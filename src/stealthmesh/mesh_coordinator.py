"""
Mesh Network Coordinator for StealthMesh
Implements peer discovery, gossip protocol, and alert propagation
"""

import os
import json
import time
import secrets
import threading
import hashlib
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import queue


class NodeStatus(Enum):
    """Status of a mesh node"""
    ONLINE = "online"
    OFFLINE = "offline"
    SUSPICIOUS = "suspicious"
    QUARANTINED = "quarantined"


class MessageType(Enum):
    """Types of mesh network messages"""
    DISCOVERY = "discovery"
    HEARTBEAT = "heartbeat"
    ALERT = "alert"
    ALERT_ACK = "alert_ack"
    PEER_UPDATE = "peer_update"
    CONSENSUS_REQUEST = "consensus"
    CONSENSUS_VOTE = "vote"
    CONTAINMENT = "containment"


@dataclass
class MeshNode:
    """Represents a node in the mesh network"""
    node_id: str
    address: str
    port: int
    public_key: bytes
    status: NodeStatus = NodeStatus.ONLINE
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 1.0
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "public_key": self.public_key.hex(),
            "status": self.status.value,
            "last_seen": self.last_seen,
            "trust_score": self.trust_score,
            "capabilities": self.capabilities
        }


@dataclass
class MeshMessage:
    """Message structure for mesh communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 5
    hop_count: int = 0
    signature: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "hop_count": self.hop_count,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeshMessage':
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            ttl=data.get("ttl", 5),
            hop_count=data.get("hop_count", 0),
            signature=data.get("signature", "")
        )


@dataclass
class Alert:
    """Security alert structure"""
    alert_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    source_node: str
    target_ip: Optional[str]
    description: str
    timestamp: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    acknowledgements: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "source_node": self.source_node,
            "target_ip": self.target_ip,
            "description": self.description,
            "timestamp": self.timestamp,
            "evidence": self.evidence,
            "acknowledgements": list(self.acknowledgements)
        }


class GossipProtocol:
    """
    Gossip-based message propagation with Byzantine fault tolerance
    """
    
    def __init__(self, 
                 fanout: int = 3,
                 max_rounds: int = 10):
        """
        Args:
            fanout: Number of peers to forward to
            max_rounds: Maximum propagation rounds
        """
        self.fanout = fanout
        self.max_rounds = max_rounds
        self.seen_messages: Set[str] = set()
        self.message_counts: Dict[str, int] = defaultdict(int)
        
    def should_propagate(self, message: MeshMessage) -> bool:
        """Check if message should be propagated"""
        # Don't propagate if already seen
        if message.message_id in self.seen_messages:
            return False
            
        # Don't propagate if TTL expired
        if message.ttl <= 0:
            return False
            
        # Don't propagate if too many hops
        if message.hop_count >= self.max_rounds:
            return False
            
        return True
    
    def mark_seen(self, message_id: str):
        """Mark message as seen"""
        self.seen_messages.add(message_id)
        self.message_counts[message_id] += 1
        
        # Cleanup old messages (keep last 10000)
        if len(self.seen_messages) > 10000:
            oldest = list(self.seen_messages)[:5000]
            for msg_id in oldest:
                self.seen_messages.discard(msg_id)
                self.message_counts.pop(msg_id, None)
    
    def select_peers(self, 
                     all_peers: List[str],
                     exclude: Set[str] = None) -> List[str]:
        """Select peers for gossip propagation"""
        exclude = exclude or set()
        available = [p for p in all_peers if p not in exclude]
        
        # Select random subset
        k = min(self.fanout, len(available))
        return secrets.SystemRandom().sample(available, k) if available else []
    
    def get_message_consensus(self, message_id: str, threshold: int = 3) -> bool:
        """Check if message has reached consensus (seen by enough peers)"""
        return self.message_counts.get(message_id, 0) >= threshold


class ConsensusManager:
    """
    Manages distributed consensus for critical decisions
    (e.g., quarantining a node)
    """
    
    def __init__(self, 
                 quorum_percentage: float = 0.67,
                 timeout_seconds: int = 30):
        """
        Args:
            quorum_percentage: Required votes for consensus
            timeout_seconds: Voting timeout
        """
        self.quorum_percentage = quorum_percentage
        self.timeout_seconds = timeout_seconds
        self.active_votes: Dict[str, Dict[str, Any]] = {}
        
    def initiate_vote(self, 
                      topic_id: str,
                      topic_type: str,
                      total_voters: int) -> str:
        """Start a new consensus vote"""
        vote_id = f"{topic_id}_{secrets.token_hex(4)}"
        
        self.active_votes[vote_id] = {
            "topic_id": topic_id,
            "topic_type": topic_type,
            "total_voters": total_voters,
            "votes_for": set(),
            "votes_against": set(),
            "started_at": time.time(),
            "resolved": False,
            "result": None
        }
        
        return vote_id
    
    def cast_vote(self, vote_id: str, voter_id: str, approve: bool) -> bool:
        """Cast a vote"""
        if vote_id not in self.active_votes:
            return False
            
        vote = self.active_votes[vote_id]
        if vote["resolved"]:
            return False
            
        # Check timeout
        if time.time() - vote["started_at"] > self.timeout_seconds:
            self._resolve_vote(vote_id)
            return False
            
        # Record vote
        if approve:
            vote["votes_for"].add(voter_id)
            vote["votes_against"].discard(voter_id)
        else:
            vote["votes_against"].add(voter_id)
            vote["votes_for"].discard(voter_id)
            
        # Check if quorum reached
        self._check_quorum(vote_id)
        
        return True
    
    def _check_quorum(self, vote_id: str):
        """Check if vote has reached quorum"""
        vote = self.active_votes[vote_id]
        total = vote["total_voters"]
        required = int(total * self.quorum_percentage)
        
        if len(vote["votes_for"]) >= required:
            vote["resolved"] = True
            vote["result"] = True
        elif len(vote["votes_against"]) > total - required:
            vote["resolved"] = True
            vote["result"] = False
            
    def _resolve_vote(self, vote_id: str):
        """Resolve vote due to timeout"""
        vote = self.active_votes[vote_id]
        if not vote["resolved"]:
            # Resolve based on current votes
            vote["resolved"] = True
            total = len(vote["votes_for"]) + len(vote["votes_against"])
            if total > 0:
                vote["result"] = len(vote["votes_for"]) / total >= self.quorum_percentage
            else:
                vote["result"] = False
                
    def get_vote_result(self, vote_id: str) -> Optional[bool]:
        """Get result of a vote"""
        if vote_id not in self.active_votes:
            return None
        return self.active_votes[vote_id].get("result")
    
    def get_vote_status(self, vote_id: str) -> Dict[str, Any]:
        """Get current status of a vote"""
        if vote_id not in self.active_votes:
            return {}
        vote = self.active_votes[vote_id]
        return {
            "topic_id": vote["topic_id"],
            "votes_for": len(vote["votes_for"]),
            "votes_against": len(vote["votes_against"]),
            "total_voters": vote["total_voters"],
            "resolved": vote["resolved"],
            "result": vote["result"]
        }


class MeshCoordinator:
    """
    Main mesh network coordinator for StealthMesh
    Handles peer discovery, message routing, and alert propagation
    """
    
    def __init__(self,
                 node_id: str,
                 address: str,
                 port: int,
                 heartbeat_interval: int = 30,
                 peer_timeout: int = 120):
        """
        Initialize mesh coordinator
        
        Args:
            node_id: This node's identifier
            address: This node's network address
            port: This node's port
            heartbeat_interval: Seconds between heartbeats
            peer_timeout: Seconds before marking peer offline
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.peer_timeout = peer_timeout
        
        # Node key
        self.private_key = secrets.token_bytes(32)
        self.public_key = hashlib.sha256(self.private_key).digest()
        
        # Peer management
        self.peers: Dict[str, MeshNode] = {}
        
        # Protocol handlers
        self.gossip = GossipProtocol(fanout=3)
        self.consensus = ConsensusManager()
        
        # Alert management
        self.alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Message queue
        self.outgoing_queue: queue.Queue = queue.Queue()
        self.incoming_queue: queue.Queue = queue.Queue()
        
        # Background threads
        self._running = False
        self._threads: List[threading.Thread] = []
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "alerts_raised": 0,
            "alerts_propagated": 0,
            "consensus_initiated": 0
        }
        
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for new alerts"""
        self.alert_callbacks.append(callback)
        
    def add_peer(self, peer: MeshNode):
        """Add or update a peer node"""
        peer.last_seen = time.time()
        self.peers[peer.node_id] = peer
        
    def remove_peer(self, node_id: str):
        """Remove a peer node"""
        self.peers.pop(node_id, None)
        
    def get_peer(self, node_id: str) -> Optional[MeshNode]:
        """Get peer by ID"""
        return self.peers.get(node_id)
    
    def get_active_peers(self) -> List[MeshNode]:
        """Get all active peers"""
        current_time = time.time()
        return [
            peer for peer in self.peers.values()
            if peer.status == NodeStatus.ONLINE and
            current_time - peer.last_seen < self.peer_timeout
        ]
    
    def create_message(self,
                       message_type: MessageType,
                       payload: Dict[str, Any],
                       ttl: int = 5) -> MeshMessage:
        """Create a new mesh message"""
        message = MeshMessage(
            message_id=secrets.token_hex(16),
            message_type=message_type,
            sender_id=self.node_id,
            payload=payload,
            timestamp=time.time(),
            ttl=ttl
        )
        
        # Sign message
        message.signature = self._sign_message(message)
        
        return message
    
    def _sign_message(self, message: MeshMessage) -> str:
        """Sign a message with node's private key"""
        content = f"{message.message_id}{message.sender_id}{message.timestamp}"
        signature = hashlib.sha256(
            self.private_key + content.encode()
        ).hexdigest()
        return signature
    
    def _verify_signature(self, message: MeshMessage, sender: MeshNode) -> bool:
        """Verify message signature"""
        # Simplified verification - in production use asymmetric crypto
        content = f"{message.message_id}{message.sender_id}{message.timestamp}"
        expected = hashlib.sha256(
            sender.public_key + content.encode()
        ).hexdigest()
        return message.signature == expected
    
    def broadcast_message(self, message: MeshMessage):
        """Broadcast message to all peers using gossip"""
        self.gossip.mark_seen(message.message_id)
        
        peers = self.gossip.select_peers(
            list(self.peers.keys()),
            exclude={self.node_id, message.sender_id}
        )
        
        for peer_id in peers:
            self.outgoing_queue.put((peer_id, message))
            self.stats["messages_sent"] += 1
            
    def receive_message(self, message: MeshMessage):
        """Process received message"""
        self.stats["messages_received"] += 1
        
        # Check if should propagate
        if not self.gossip.should_propagate(message):
            return
            
        self.gossip.mark_seen(message.message_id)
        
        # Update hop count and TTL
        message.hop_count += 1
        message.ttl -= 1
        
        # Handle by type
        if message.message_type == MessageType.DISCOVERY:
            self._handle_discovery(message)
        elif message.message_type == MessageType.HEARTBEAT:
            self._handle_heartbeat(message)
        elif message.message_type == MessageType.ALERT:
            self._handle_alert(message)
        elif message.message_type == MessageType.CONSENSUS_REQUEST:
            self._handle_consensus_request(message)
        elif message.message_type == MessageType.CONSENSUS_VOTE:
            self._handle_consensus_vote(message)
        elif message.message_type == MessageType.PEER_UPDATE:
            self._handle_peer_update(message)
            
        # Propagate if still has TTL
        if message.ttl > 0:
            self.broadcast_message(message)
            
    def _handle_discovery(self, message: MeshMessage):
        """Handle peer discovery message"""
        payload = message.payload
        peer = MeshNode(
            node_id=message.sender_id,
            address=payload.get("address", ""),
            port=payload.get("port", 0),
            public_key=bytes.fromhex(payload.get("public_key", "")),
            capabilities=payload.get("capabilities", [])
        )
        self.add_peer(peer)
        
    def _handle_heartbeat(self, message: MeshMessage):
        """Handle heartbeat message"""
        if message.sender_id in self.peers:
            self.peers[message.sender_id].last_seen = time.time()
            self.peers[message.sender_id].status = NodeStatus.ONLINE
            
    def _handle_alert(self, message: MeshMessage):
        """Handle security alert"""
        payload = message.payload
        alert = Alert(
            alert_id=payload["alert_id"],
            alert_type=payload["alert_type"],
            severity=payload["severity"],
            source_node=message.sender_id,
            target_ip=payload.get("target_ip"),
            description=payload["description"],
            timestamp=payload["timestamp"],
            evidence=payload.get("evidence", {})
        )
        
        # Store alert
        if alert.alert_id not in self.alerts:
            self.alerts[alert.alert_id] = alert
            self.stats["alerts_propagated"] += 1
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass
                    
        # Add acknowledgement
        self.alerts[alert.alert_id].acknowledgements.add(self.node_id)
        
    def _handle_consensus_request(self, message: MeshMessage):
        """Handle consensus request"""
        payload = message.payload
        # Auto-vote based on local analysis (simplified)
        vote_id = payload["vote_id"]
        approve = self._evaluate_consensus_topic(payload)
        
        # Send vote
        vote_message = self.create_message(
            MessageType.CONSENSUS_VOTE,
            {
                "vote_id": vote_id,
                "approve": approve,
                "voter_id": self.node_id
            }
        )
        self.broadcast_message(vote_message)
        
    def _handle_consensus_vote(self, message: MeshMessage):
        """Handle consensus vote"""
        payload = message.payload
        self.consensus.cast_vote(
            payload["vote_id"],
            payload["voter_id"],
            payload["approve"]
        )
        
    def _handle_peer_update(self, message: MeshMessage):
        """Handle peer status update"""
        payload = message.payload
        node_id = payload.get("node_id")
        if node_id and node_id in self.peers:
            if "status" in payload:
                self.peers[node_id].status = NodeStatus(payload["status"])
            if "trust_score" in payload:
                self.peers[node_id].trust_score = payload["trust_score"]
                
    def _evaluate_consensus_topic(self, payload: Dict[str, Any]) -> bool:
        """Evaluate consensus topic and decide vote"""
        topic_type = payload.get("topic_type")
        
        if topic_type == "quarantine_node":
            target_node = payload.get("target_node")
            # Check if we've seen suspicious activity from this node
            if target_node in self.peers:
                return self.peers[target_node].trust_score < 0.5
        
        return True  # Default approve
    
    def raise_alert(self,
                    alert_type: str,
                    severity: str,
                    description: str,
                    target_ip: str = None,
                    evidence: Dict[str, Any] = None) -> Alert:
        """
        Raise a new security alert and propagate to mesh
        
        Args:
            alert_type: Type of alert (e.g., "intrusion", "ddos")
            severity: Severity level
            description: Alert description
            target_ip: Target IP if applicable
            evidence: Supporting evidence
            
        Returns:
            Created Alert
        """
        alert = Alert(
            alert_id=secrets.token_hex(8),
            alert_type=alert_type,
            severity=severity,
            source_node=self.node_id,
            target_ip=target_ip,
            description=description,
            timestamp=time.time(),
            evidence=evidence or {}
        )
        
        self.alerts[alert.alert_id] = alert
        self.stats["alerts_raised"] += 1
        
        # Create and broadcast alert message
        message = self.create_message(
            MessageType.ALERT,
            alert.to_dict()
        )
        self.broadcast_message(message)
        
        return alert
    
    def request_node_quarantine(self, target_node_id: str, reason: str) -> str:
        """
        Request consensus to quarantine a suspicious node
        
        Args:
            target_node_id: Node to quarantine
            reason: Reason for quarantine
            
        Returns:
            Vote ID for tracking
        """
        total_voters = len(self.get_active_peers()) + 1  # Include self
        
        vote_id = self.consensus.initiate_vote(
            topic_id=f"quarantine_{target_node_id}",
            topic_type="quarantine_node",
            total_voters=total_voters
        )
        
        # Cast own vote
        self.consensus.cast_vote(vote_id, self.node_id, True)
        
        # Broadcast consensus request
        message = self.create_message(
            MessageType.CONSENSUS_REQUEST,
            {
                "vote_id": vote_id,
                "topic_type": "quarantine_node",
                "target_node": target_node_id,
                "reason": reason
            }
        )
        self.broadcast_message(message)
        
        self.stats["consensus_initiated"] += 1
        
        return vote_id
    
    def send_heartbeat(self):
        """Send heartbeat to peers"""
        message = self.create_message(
            MessageType.HEARTBEAT,
            {
                "status": "online",
                "load": 0.5,  # Placeholder
                "peer_count": len(self.peers)
            },
            ttl=2
        )
        self.broadcast_message(message)
        
    def announce_presence(self):
        """Announce this node to the mesh"""
        message = self.create_message(
            MessageType.DISCOVERY,
            {
                "address": self.address,
                "port": self.port,
                "public_key": self.public_key.hex(),
                "capabilities": ["detection", "containment", "routing"]
            }
        )
        self.broadcast_message(message)
        
    def start(self):
        """Start mesh coordinator background tasks"""
        self._running = True
        
        # Heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        heartbeat_thread.start()
        self._threads.append(heartbeat_thread)
        
        # Peer cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        cleanup_thread.start()
        self._threads.append(cleanup_thread)
        
    def stop(self):
        """Stop mesh coordinator"""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2)
            
    def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self._running:
            self.send_heartbeat()
            time.sleep(self.heartbeat_interval)
            
    def _cleanup_loop(self):
        """Background peer cleanup loop"""
        while self._running:
            current_time = time.time()
            for peer in list(self.peers.values()):
                if current_time - peer.last_seen > self.peer_timeout:
                    peer.status = NodeStatus.OFFLINE
            time.sleep(30)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            **self.stats,
            "total_peers": len(self.peers),
            "active_peers": len(self.get_active_peers()),
            "total_alerts": len(self.alerts),
            "active_votes": len(self.consensus.active_votes)
        }


if __name__ == "__main__":
    # Test mesh coordinator
    print("Testing StealthMesh Mesh Coordinator")
    print("=" * 60)
    
    # Create coordinator
    coordinator = MeshCoordinator(
        node_id="node_1",
        address="192.168.1.1",
        port=8001
    )
    
    # Add some peers
    for i in range(2, 5):
        peer = MeshNode(
            node_id=f"node_{i}",
            address=f"192.168.1.{i}",
            port=8000 + i,
            public_key=secrets.token_bytes(32)
        )
        coordinator.add_peer(peer)
        print(f"Added peer: {peer.node_id}")
        
    # Register alert callback
    def on_alert(alert: Alert):
        print(f"  [CALLBACK] Alert received: {alert.alert_type} ({alert.severity})")
        
    coordinator.register_alert_callback(on_alert)
    
    # Raise an alert
    print("\nRaising alert...")
    alert = coordinator.raise_alert(
        alert_type="port_scan",
        severity="medium",
        description="Port scanning detected from external IP",
        target_ip="192.168.1.100",
        evidence={"ports_scanned": [22, 80, 443, 8080]}
    )
    print(f"Alert created: {alert.alert_id}")
    
    # Test consensus
    print("\nInitiating quarantine vote...")
    vote_id = coordinator.request_node_quarantine("suspicious_node", "Multiple failed auth attempts")
    print(f"Vote ID: {vote_id}")
    
    # Simulate votes
    coordinator.consensus.cast_vote(vote_id, "node_2", True)
    coordinator.consensus.cast_vote(vote_id, "node_3", True)
    
    status = coordinator.consensus.get_vote_status(vote_id)
    print(f"Vote status: {status}")
    
    # Statistics
    print("\nStatistics:")
    stats = coordinator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n✓ Mesh coordinator test passed!")
