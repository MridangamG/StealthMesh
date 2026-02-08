"""
StealthMesh Node - Unified Defense Node
Integrates all StealthMesh modules into a single defense unit
"""

import os
import sys
import time
import json
import secrets
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .stealth_comm import StealthCommunication, CamouflageType, generate_node_key
from .decoy_routing import DecoyRouter, RouteNode, RouteType
from .mesh_coordinator import MeshCoordinator, MeshNode, NodeStatus, Alert
from .threat_detector import ThreatDetector, ThreatDetection, NetworkFlow, ThreatLevel
from .micro_containment import MicroContainment, ContainmentRule, ThreatSeverity
from .adaptive_mtd import AdaptiveMTD, MutationEvent


class NodeMode(Enum):
    """Operating modes for StealthMesh node"""
    NORMAL = "normal"           # Standard operation
    ALERT = "alert"             # Elevated threat awareness
    DEFENSE = "defense"         # Active defense mode
    LOCKDOWN = "lockdown"       # Maximum security


@dataclass
class NodeConfig:
    """Configuration for StealthMesh node"""
    node_id: str
    address: str
    port: int
    
    # Feature toggles
    enable_stealth_comm: bool = True
    enable_decoy_routing: bool = True
    enable_mesh_network: bool = True
    enable_threat_detection: bool = True
    enable_micro_containment: bool = True
    enable_adaptive_mtd: bool = True
    
    # Parameters
    detection_model: str = "xgboost"
    containment_dry_run: bool = True
    mtd_mutation_interval: int = 300
    heartbeat_interval: int = 30
    
    # Stealth settings
    camouflage_type: str = "http"
    key_rotation_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "features": {
                "stealth_comm": self.enable_stealth_comm,
                "decoy_routing": self.enable_decoy_routing,
                "mesh_network": self.enable_mesh_network,
                "threat_detection": self.enable_threat_detection,
                "micro_containment": self.enable_micro_containment,
                "adaptive_mtd": self.enable_adaptive_mtd
            }
        }


class StealthMeshNode:
    """
    Complete StealthMesh defense node integrating all modules
    
    This is the main class that combines:
    - Stealth Communication (polymorphic encryption, packet camouflage)
    - Decoy Routing (dynamic paths, fake traffic)
    - Mesh Coordination (peer discovery, alert propagation)
    - Threat Detection (ML-based attack classification)
    - Micro-Containment (autonomous breach isolation)
    - Adaptive MTD (attack surface mutation)
    """
    
    def __init__(self, config: NodeConfig):
        """
        Initialize StealthMesh node
        
        Args:
            config: Node configuration
        """
        self.config = config
        self.node_id = config.node_id
        self.mode = NodeMode.NORMAL
        
        # Generate node key
        self.node_key = generate_node_key()
        
        # Initialize modules
        self._init_modules()
        
        # Event handlers
        self._setup_event_handlers()
        
        # State tracking
        self.is_running = False
        self.start_time: float = 0.0
        self.last_threat_time: float = 0.0
        
        # Statistics
        self.stats = {
            "flows_analyzed": 0,
            "attacks_detected": 0,
            "alerts_sent": 0,
            "alerts_received": 0,
            "containments_applied": 0,
            "mutations_performed": 0,
            "messages_sent": 0,
            "messages_received": 0
        }
        
        print(f"[StealthMesh] Node {self.node_id} initialized")
        
    def _init_modules(self):
        """Initialize all defense modules"""
        
        # 1. Stealth Communication
        if self.config.enable_stealth_comm:
            camouflage = CamouflageType(self.config.camouflage_type)
            self.stealth_comm = StealthCommunication(
                master_key=self.node_key,
                camouflage_type=camouflage,
                rotation_interval=self.config.key_rotation_interval
            )
            print(f"  ✓ Stealth Communication ({camouflage.value})")
        else:
            self.stealth_comm = None
            
        # 2. Decoy Routing
        if self.config.enable_decoy_routing:
            self.decoy_router = DecoyRouter(
                node_id=self.node_id,
                randomization_factor=0.3,
                decoy_traffic_rate=0.5
            )
            print("  ✓ Decoy Routing")
        else:
            self.decoy_router = None
            
        # 3. Mesh Coordinator
        if self.config.enable_mesh_network:
            self.mesh = MeshCoordinator(
                node_id=self.node_id,
                address=self.config.address,
                port=self.config.port,
                heartbeat_interval=self.config.heartbeat_interval
            )
            print("  ✓ Mesh Coordinator")
        else:
            self.mesh = None
            
        # 4. Threat Detector
        if self.config.enable_threat_detection:
            try:
                self.detector = ThreatDetector(
                    model_name=self.config.detection_model,
                    confidence_threshold=0.7
                )
                print(f"  ✓ Threat Detector ({self.config.detection_model})")
            except Exception as e:
                print(f"  ✗ Threat Detector failed: {e}")
                self.detector = None
        else:
            self.detector = None
            
        # 5. Micro-Containment
        if self.config.enable_micro_containment:
            self.containment = MicroContainment(
                dry_run=self.config.containment_dry_run,
                auto_contain=True
            )
            print(f"  ✓ Micro-Containment (dry_run={self.config.containment_dry_run})")
        else:
            self.containment = None
            
        # 6. Adaptive MTD
        if self.config.enable_adaptive_mtd:
            self.mtd = AdaptiveMTD(
                mutation_interval=self.config.mtd_mutation_interval,
                enable_port_hopping=True,
                enable_decoys=True,
                dry_run=True
            )
            print("  ✓ Adaptive MTD")
        else:
            self.mtd = None
            
    def _setup_event_handlers(self):
        """Set up event handlers between modules"""
        
        # When threat detected -> raise mesh alert + containment
        if self.detector:
            self.detector.register_callback(self._on_threat_detected)
            
        # When alert received from mesh -> update MTD threat level
        if self.mesh:
            self.mesh.register_alert_callback(self._on_alert_received)
            
        # When containment applied -> notify mesh
        if self.containment:
            self.containment.register_callback(self._on_containment_applied)
            
        # When MTD mutation -> update mesh peers
        if self.mtd:
            self.mtd.register_callback(self._on_mtd_mutation)
            
    def _on_threat_detected(self, detection: ThreatDetection):
        """Handle threat detection"""
        self.stats["attacks_detected"] += 1
        self.last_threat_time = time.time()
        
        print(f"[THREAT] {detection.attack_type.value} detected from {detection.source_ip}")
        print(f"         Confidence: {detection.confidence:.2f}, Level: {detection.threat_level.value}")
        
        # Update mode based on threat
        self._update_mode(detection.threat_level)
        
        # Report to containment
        if self.containment and detection.source_ip:
            severity = self._threat_to_severity(detection.threat_level)
            self.containment.report_breach(
                source_ip=detection.source_ip,
                attack_type=detection.attack_type.value,
                target_ip=detection.dest_ip,
                evidence=detection.features,
                severity=severity
            )
            
        # Raise mesh alert
        if self.mesh:
            self.mesh.raise_alert(
                alert_type=detection.attack_type.value,
                severity=detection.threat_level.value,
                description=f"Attack detected with {detection.confidence:.0%} confidence",
                target_ip=detection.dest_ip,
                evidence={
                    "detection_id": detection.detection_id,
                    "source_ip": detection.source_ip,
                    "model": detection.model_used
                }
            )
            self.stats["alerts_sent"] += 1
            
        # Update MTD threat level
        if self.mtd:
            threat_value = {"none": 0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            self.mtd.update_threat_level(threat_value.get(detection.threat_level.value, 0.5))
            self.mtd.report_attack({
                "source_ip": detection.source_ip,
                "attack_type": detection.attack_type.value
            })
            
    def _on_alert_received(self, alert: Alert):
        """Handle alert received from mesh network"""
        self.stats["alerts_received"] += 1
        
        print(f"[MESH ALERT] {alert.alert_type} from {alert.source_node}")
        print(f"             Severity: {alert.severity}, Target: {alert.target_ip}")
        
        # Update MTD based on mesh intelligence
        if self.mtd:
            threat_value = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            current = self.mtd.threat_level
            received = threat_value.get(alert.severity, 0.5)
            # Weighted average with current threat level
            self.mtd.update_threat_level(current * 0.7 + received * 0.3)
            
        # Consider preemptive containment
        if self.containment and alert.target_ip:
            # Only if high severity
            if alert.severity in ["high", "critical"]:
                self.containment.report_breach(
                    source_ip=alert.evidence.get("source_ip", "unknown"),
                    attack_type=alert.alert_type,
                    target_ip=alert.target_ip,
                    evidence=alert.evidence,
                    severity=ThreatSeverity.MEDIUM  # Reduce severity for mesh alerts
                )
                
    def _on_containment_applied(self, rule: ContainmentRule):
        """Handle containment rule application"""
        self.stats["containments_applied"] += 1
        
        print(f"[CONTAINMENT] {rule.action.value} applied to {rule.target}")
        
        # Broadcast containment to mesh
        if self.mesh:
            from .mesh_coordinator import MessageType
            message = self.mesh.create_message(
                MessageType.CONTAINMENT,
                {
                    "action": rule.action.value,
                    "target": rule.target,
                    "severity": rule.severity.name,
                    "reason": rule.reason
                }
            )
            self.mesh.broadcast_message(message)
            
    def _on_mtd_mutation(self, event: MutationEvent):
        """Handle MTD mutation event"""
        self.stats["mutations_performed"] += 1
        
        print(f"[MTD] {event.mutation_type.value}: {event.old_value} -> {event.new_value}")
        
        # Notify mesh of service changes
        if self.mesh and event.service_id:
            from .mesh_coordinator import MessageType
            message = self.mesh.create_message(
                MessageType.PEER_UPDATE,
                {
                    "service_mutation": {
                        "service_id": event.service_id,
                        "mutation_type": event.mutation_type.value,
                        "new_value": str(event.new_value)
                    }
                }
            )
            self.mesh.broadcast_message(message)
            
    def _threat_to_severity(self, threat_level: ThreatLevel) -> ThreatSeverity:
        """Convert threat level to containment severity"""
        mapping = {
            ThreatLevel.NONE: ThreatSeverity.LOW,
            ThreatLevel.LOW: ThreatSeverity.LOW,
            ThreatLevel.MEDIUM: ThreatSeverity.MEDIUM,
            ThreatLevel.HIGH: ThreatSeverity.HIGH,
            ThreatLevel.CRITICAL: ThreatSeverity.CRITICAL
        }
        return mapping.get(threat_level, ThreatSeverity.MEDIUM)
    
    def _update_mode(self, threat_level: ThreatLevel):
        """Update node operating mode based on threat"""
        if threat_level == ThreatLevel.CRITICAL:
            self.mode = NodeMode.LOCKDOWN
        elif threat_level == ThreatLevel.HIGH:
            self.mode = NodeMode.DEFENSE
        elif threat_level == ThreatLevel.MEDIUM:
            self.mode = NodeMode.ALERT
        else:
            # Gradually return to normal
            if time.time() - self.last_threat_time > 300:  # 5 minutes
                self.mode = NodeMode.NORMAL
                
    def add_peer(self, 
                 node_id: str, 
                 address: str, 
                 port: int,
                 public_key: bytes = None):
        """
        Add a peer node to the mesh network
        
        Args:
            node_id: Peer's node ID
            address: Peer's network address
            port: Peer's port
            public_key: Peer's public key
        """
        if self.mesh:
            peer = MeshNode(
                node_id=node_id,
                address=address,
                port=port,
                public_key=public_key or secrets.token_bytes(32)
            )
            self.mesh.add_peer(peer)
            
        if self.decoy_router:
            route_node = RouteNode(
                node_id=node_id,
                address=address,
                port=port,
                public_key=public_key or secrets.token_bytes(32)
            )
            self.decoy_router.register_peer(route_node)
            
        print(f"[PEER] Added {node_id} at {address}:{port}")
        
    def analyze_flow(self, flow: NetworkFlow) -> Optional[ThreatDetection]:
        """
        Analyze a network flow for threats
        
        Args:
            flow: Network flow data
            
        Returns:
            ThreatDetection result or None
        """
        if not self.detector:
            return None
            
        self.stats["flows_analyzed"] += 1
        return self.detector.detect(flow)
    
    def analyze_flows_batch(self, flows: List[NetworkFlow]) -> List[ThreatDetection]:
        """
        Analyze multiple flows in batch
        
        Args:
            flows: List of network flows
            
        Returns:
            List of detection results
        """
        if not self.detector:
            return []
            
        self.stats["flows_analyzed"] += len(flows)
        return self.detector.detect_batch(flows)
    
    def send_secure_message(self, 
                            destination: str, 
                            data: Dict[str, Any]) -> bool:
        """
        Send a message securely through the mesh
        
        Args:
            destination: Target node ID
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.stealth_comm:
            return False
            
        # Create stealth packet
        packet = self.stealth_comm.create_stealth_packet(data)
        serialized = self.stealth_comm.serialize_packet(packet)
        
        # Route through decoy router if available
        if self.decoy_router:
            route, decoys = self.decoy_router.route_packet(serialized, destination)
            # In real implementation, would send through route
            
        self.stats["messages_sent"] += 1
        return True
    
    def register_service(self, 
                         name: str, 
                         port: int, 
                         is_critical: bool = False):
        """
        Register a service for MTD protection
        
        Args:
            name: Service name
            port: Service port
            is_critical: If service is critical
        """
        if self.mtd:
            self.mtd.register_service(name, port, is_critical=is_critical)
            print(f"[SERVICE] Registered {name} on port {port}")
            
    def whitelist_ip(self, ip: str):
        """Add IP to containment whitelist"""
        if self.containment:
            self.containment.policy.add_whitelist(ip)
            print(f"[WHITELIST] Added {ip}")
            
    def blacklist_ip(self, ip: str):
        """Add IP to permanent blacklist"""
        if self.containment:
            self.containment.policy.add_blacklist(ip)
            print(f"[BLACKLIST] Added {ip}")
            
    def start(self):
        """Start all StealthMesh modules"""
        print(f"\n[StealthMesh] Starting node {self.node_id}...")
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start mesh coordinator
        if self.mesh:
            self.mesh.start()
            self.mesh.announce_presence()
            
        # Start containment background tasks
        if self.containment:
            self.containment.start()
            
        # Start MTD
        if self.mtd:
            self.mtd.start()
            # Spawn initial decoys
            self.mtd.spawn_decoys(count=3)
            
        # Start decoy traffic
        if self.decoy_router:
            self.decoy_router.start_decoy_traffic()
            
        print(f"[StealthMesh] Node {self.node_id} is now active in {self.mode.value} mode")
        
    def stop(self):
        """Stop all StealthMesh modules"""
        print(f"\n[StealthMesh] Stopping node {self.node_id}...")
        
        self.is_running = False
        
        if self.decoy_router:
            self.decoy_router.stop_decoy_traffic()
            
        if self.mtd:
            self.mtd.stop()
            
        if self.containment:
            self.containment.stop()
            
        if self.mesh:
            self.mesh.stop()
            
        print(f"[StealthMesh] Node {self.node_id} stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status"""
        uptime = time.time() - self.start_time if self.is_running else 0
        
        status = {
            "node_id": self.node_id,
            "mode": self.mode.value,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "config": self.config.to_dict(),
            "stats": self.stats.copy()
        }
        
        # Add module statistics
        if self.detector:
            status["detector"] = self.detector.get_statistics()
            
        if self.containment:
            status["containment"] = self.containment.get_statistics()
            
        if self.mtd:
            status["mtd"] = self.mtd.get_statistics()
            
        if self.mesh:
            status["mesh"] = self.mesh.get_statistics()
            
        if self.decoy_router:
            status["routing"] = self.decoy_router.get_statistics()
            
        return status
    
    def print_status(self):
        """Print formatted status report"""
        status = self.get_status()
        
        print("\n" + "=" * 70)
        print(f"STEALTHMESH NODE STATUS: {self.node_id}")
        print("=" * 70)
        print(f"Mode: {status['mode'].upper()}")
        print(f"Running: {status['is_running']}")
        print(f"Uptime: {status['uptime_seconds']:.0f} seconds")
        
        print("\n--- Statistics ---")
        for key, value in status['stats'].items():
            print(f"  {key}: {value}")
            
        if 'detector' in status:
            print("\n--- Threat Detection ---")
            for key, value in status['detector'].items():
                print(f"  {key}: {value}")
                
        if 'containment' in status:
            print("\n--- Containment ---")
            for key, value in status['containment'].items():
                print(f"  {key}: {value}")
                
        if 'mtd' in status:
            print("\n--- Moving Target Defense ---")
            for key, value in status['mtd'].items():
                print(f"  {key}: {value}")
                
        if 'mesh' in status:
            print("\n--- Mesh Network ---")
            for key, value in status['mesh'].items():
                print(f"  {key}: {value}")
                
        print("=" * 70)


def create_stealthmesh_node(
    node_id: str = None,
    address: str = "127.0.0.1",
    port: int = 8000,
    **kwargs
) -> StealthMeshNode:
    """
    Factory function to create a StealthMesh node
    
    Args:
        node_id: Node identifier (generated if not provided)
        address: Network address
        port: Network port
        **kwargs: Additional config options
        
    Returns:
        Configured StealthMeshNode
    """
    if node_id is None:
        node_id = f"node_{secrets.token_hex(4)}"
        
    config = NodeConfig(
        node_id=node_id,
        address=address,
        port=port,
        **kwargs
    )
    
    return StealthMeshNode(config)


if __name__ == "__main__":
    # Test StealthMesh Node
    print("Testing StealthMesh Node")
    print("=" * 70)
    
    # Create node
    node = create_stealthmesh_node(
        node_id="test_node_1",
        address="192.168.1.10",
        port=8001,
        detection_model="randomforest",  # Use RandomForest for testing
        containment_dry_run=True
    )
    
    # Add some peers
    node.add_peer("peer_1", "192.168.1.20", 8002)
    node.add_peer("peer_2", "192.168.1.30", 8003)
    
    # Register services for MTD
    node.register_service("web_server", 80, is_critical=True)
    node.register_service("api_server", 3000)
    
    # Whitelist trusted IP
    node.whitelist_ip("192.168.1.1")  # Gateway
    
    # Start node
    node.start()
    
    # Simulate flow analysis
    print("\n--- Simulating Traffic Analysis ---")
    
    # Normal flow
    normal_flow = NetworkFlow(
        flow_id="flow_normal_1",
        src_ip="192.168.1.100",
        dst_ip="8.8.8.8",
        src_port=54321,
        dst_port=443,
        protocol=6,
        timestamp=time.time(),
        flow_duration=1000,
        total_fwd_packets=10,
        total_bwd_packets=8,
        flow_bytes_per_sec=2500
    )
    
    result = node.analyze_flow(normal_flow)
    if result:
        print(f"Normal flow: {'ATTACK' if result.is_attack else 'BENIGN'}")
    
    # Suspicious flow
    suspicious_flow = NetworkFlow(
        flow_id="flow_suspicious_1",
        src_ip="10.0.0.100",
        dst_ip="192.168.1.10",
        src_port=12345,
        dst_port=22,
        protocol=6,
        timestamp=time.time(),
        flow_duration=10,
        total_fwd_packets=1000,
        total_bwd_packets=0,
        flow_bytes_per_sec=1000000,
        syn_flag_count=1000
    )
    
    result = node.analyze_flow(suspicious_flow)
    if result:
        print(f"Suspicious flow: {'ATTACK' if result.is_attack else 'BENIGN'}")
    
    # Wait a bit
    time.sleep(2)
    
    # Print status
    node.print_status()
    
    # Stop node
    node.stop()
    
    print("\n✓ StealthMesh Node test completed!")
