"""
Decoy Routing Module for StealthMesh
Implements dynamic path selection, fake traffic injection, and deception coordination
"""

import os
import json
import random
import secrets
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib


class RouteType(Enum):
    """Types of routes in the mesh network"""
    DIRECT = "direct"           # Direct peer-to-peer
    MULTI_HOP = "multi_hop"     # Through multiple nodes
    DECOY = "decoy"             # Fake route for deception
    ONION = "onion"             # Layered encryption route


class TrafficType(Enum):
    """Types of traffic for cover generation"""
    HEARTBEAT = "heartbeat"
    FAKE_ALERT = "fake_alert"
    PADDING = "padding"
    CHAFF = "chaff"


@dataclass
class RouteNode:
    """Represents a node in a route path"""
    node_id: str
    address: str
    port: int
    public_key: bytes
    latency_ms: float = 0.0
    trust_score: float = 1.0
    
    
@dataclass
class Route:
    """A complete route through the mesh network"""
    route_id: str
    route_type: RouteType
    path: List[RouteNode]
    created_at: float
    expires_at: float
    is_active: bool = True
    usage_count: int = 0
    
    
@dataclass
class DecoyPacket:
    """Fake traffic packet for deception"""
    packet_id: str
    traffic_type: TrafficType
    payload: bytes
    destination: str
    created_at: float
    ttl: int = 3


class RouteSelector:
    """
    Intelligent route selection with randomization
    to prevent traffic analysis
    """
    
    def __init__(self, randomization_factor: float = 0.3):
        """
        Args:
            randomization_factor: How much randomness to add (0-1)
        """
        self.randomization_factor = randomization_factor
        self.route_history: Dict[str, List[str]] = defaultdict(list)
        
    def select_route(self, 
                     available_routes: List[Route],
                     destination: str,
                     prefer_multi_hop: bool = True) -> Route:
        """
        Select optimal route with randomization
        
        Args:
            available_routes: List of available routes
            destination: Target destination
            prefer_multi_hop: Whether to prefer multi-hop routes
            
        Returns:
            Selected route
        """
        if not available_routes:
            raise ValueError("No routes available")
            
        # Filter active routes
        active_routes = [r for r in available_routes if r.is_active]
        if not active_routes:
            raise ValueError("No active routes available")
            
        # Score routes
        scored_routes = []
        for route in active_routes:
            score = self._calculate_route_score(route, destination, prefer_multi_hop)
            scored_routes.append((route, score))
            
        # Sort by score (higher is better)
        scored_routes.sort(key=lambda x: x[1], reverse=True)
        
        # Apply randomization - sometimes pick non-optimal route
        if random.random() < self.randomization_factor and len(scored_routes) > 1:
            # Pick from top 3 routes randomly
            top_routes = scored_routes[:min(3, len(scored_routes))]
            selected = random.choice(top_routes)[0]
        else:
            selected = scored_routes[0][0]
            
        # Record usage
        self.route_history[destination].append(selected.route_id)
        
        return selected
    
    def _calculate_route_score(self, 
                                route: Route, 
                                destination: str,
                                prefer_multi_hop: bool) -> float:
        """Calculate route quality score"""
        score = 100.0
        
        # Penalize routes used recently (avoid patterns)
        recent_usage = self.route_history.get(destination, [])[-10:]
        if route.route_id in recent_usage:
            score -= 20 * recent_usage.count(route.route_id)
            
        # Consider path length
        if prefer_multi_hop:
            # Prefer 2-4 hop routes
            if 2 <= len(route.path) <= 4:
                score += 10
            elif len(route.path) > 4:
                score -= 5 * (len(route.path) - 4)
        else:
            # Prefer shorter routes
            score -= 5 * len(route.path)
            
        # Consider trust scores
        avg_trust = sum(n.trust_score for n in route.path) / len(route.path)
        score += avg_trust * 20
        
        # Consider latency
        total_latency = sum(n.latency_ms for n in route.path)
        score -= total_latency / 100
        
        # Penalize heavily used routes
        score -= route.usage_count * 0.5
        
        # Bonus for decoy routes in high-threat scenarios
        if route.route_type == RouteType.DECOY:
            score += 15
            
        return score


class DecoyTrafficGenerator:
    """
    Generates fake traffic to confuse attackers
    and mask real communications
    """
    
    def __init__(self, 
                 traffic_rate: float = 1.0,
                 variance: float = 0.5):
        """
        Args:
            traffic_rate: Base packets per second
            variance: Random variance in timing
        """
        self.traffic_rate = traffic_rate
        self.variance = variance
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[callable] = []
        
    def add_callback(self, callback: callable):
        """Add callback for generated decoy packets"""
        self._callbacks.append(callback)
        
    def generate_decoy_packet(self, 
                               traffic_type: TrafficType = None,
                               destination: str = None) -> DecoyPacket:
        """
        Generate a single decoy packet
        
        Args:
            traffic_type: Type of fake traffic
            destination: Target destination (random if not specified)
            
        Returns:
            DecoyPacket
        """
        if traffic_type is None:
            traffic_type = random.choice(list(TrafficType))
            
        # Generate fake payload based on type
        if traffic_type == TrafficType.HEARTBEAT:
            payload = self._generate_heartbeat()
        elif traffic_type == TrafficType.FAKE_ALERT:
            payload = self._generate_fake_alert()
        elif traffic_type == TrafficType.PADDING:
            payload = self._generate_padding()
        else:  # CHAFF
            payload = self._generate_chaff()
            
        return DecoyPacket(
            packet_id=secrets.token_hex(8),
            traffic_type=traffic_type,
            payload=payload,
            destination=destination or f"node_{random.randint(1, 100)}",
            created_at=time.time(),
            ttl=random.randint(1, 5)
        )
    
    def _generate_heartbeat(self) -> bytes:
        """Generate fake heartbeat message"""
        heartbeat = {
            "type": "heartbeat",
            "node_id": f"node_{random.randint(1, 100)}",
            "timestamp": time.time(),
            "status": random.choice(["ok", "busy", "idle"]),
            "load": random.uniform(0, 1)
        }
        return json.dumps(heartbeat).encode()
    
    def _generate_fake_alert(self) -> bytes:
        """Generate fake security alert"""
        alert_types = ["port_scan", "brute_force", "anomaly", "ddos_attempt"]
        alert = {
            "type": "alert",
            "alert_type": random.choice(alert_types),
            "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
            "severity": random.choice(["low", "medium", "high"]),
            "timestamp": time.time(),
            "is_decoy": True  # Marked internally, stripped before transmission
        }
        return json.dumps(alert).encode()
    
    def _generate_padding(self) -> bytes:
        """Generate random padding data"""
        size = random.randint(64, 512)
        return secrets.token_bytes(size)
    
    def _generate_chaff(self) -> bytes:
        """Generate chaff data (looks like encrypted traffic)"""
        size = random.randint(128, 1024)
        chaff = secrets.token_bytes(size)
        # Add some structure to look like real encrypted data
        header = b"STLTH" + secrets.token_bytes(3)
        return header + chaff
    
    def start_continuous_generation(self, destinations: List[str] = None):
        """Start continuous decoy traffic generation in background"""
        self.is_running = True
        self._thread = threading.Thread(
            target=self._generation_loop,
            args=(destinations,),
            daemon=True
        )
        self._thread.start()
        
    def stop_generation(self):
        """Stop decoy traffic generation"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
            
    def _generation_loop(self, destinations: List[str]):
        """Background generation loop"""
        while self.is_running:
            # Generate packet
            dest = random.choice(destinations) if destinations else None
            packet = self.generate_decoy_packet(destination=dest)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(packet)
                except Exception:
                    pass
                    
            # Sleep with variance
            base_interval = 1.0 / self.traffic_rate
            sleep_time = base_interval * (1 + random.uniform(-self.variance, self.variance))
            time.sleep(max(0.1, sleep_time))


class DecoyRouter:
    """
    Main decoy routing interface for StealthMesh
    Combines route selection with decoy traffic generation
    """
    
    def __init__(self, 
                 node_id: str,
                 randomization_factor: float = 0.3,
                 decoy_traffic_rate: float = 0.5):
        """
        Initialize decoy router
        
        Args:
            node_id: This node's identifier
            randomization_factor: Route selection randomness
            decoy_traffic_rate: Fake traffic rate (packets/sec)
        """
        self.node_id = node_id
        self.route_selector = RouteSelector(randomization_factor)
        self.decoy_generator = DecoyTrafficGenerator(decoy_traffic_rate)
        
        # Route management
        self.routes: Dict[str, Route] = {}
        self.peer_nodes: Dict[str, RouteNode] = {}
        
        # Statistics
        self.stats = {
            "packets_routed": 0,
            "decoy_packets_sent": 0,
            "routes_created": 0,
            "routes_expired": 0
        }
        
    def register_peer(self, peer: RouteNode):
        """Register a peer node for routing"""
        self.peer_nodes[peer.node_id] = peer
        
    def create_route(self,
                     destination: str,
                     route_type: RouteType = RouteType.MULTI_HOP,
                     max_hops: int = 4,
                     ttl_seconds: int = 3600) -> Route:
        """
        Create a new route to destination
        
        Args:
            destination: Target node ID
            route_type: Type of route
            max_hops: Maximum nodes in path
            ttl_seconds: Route validity period
            
        Returns:
            Created Route
        """
        # Select intermediate nodes
        available_nodes = [
            n for n in self.peer_nodes.values() 
            if n.node_id != destination and n.node_id != self.node_id
        ]
        
        if route_type == RouteType.DIRECT:
            # Direct route
            path = [self.peer_nodes.get(destination)]
        elif route_type == RouteType.MULTI_HOP:
            # Multi-hop route
            num_hops = min(max_hops - 1, len(available_nodes))
            intermediate = random.sample(available_nodes, min(num_hops, len(available_nodes)))
            
            # Add destination at end
            if destination in self.peer_nodes:
                path = intermediate + [self.peer_nodes[destination]]
            else:
                path = intermediate
        elif route_type == RouteType.DECOY:
            # Decoy route - includes fake nodes
            num_real = min(2, len(available_nodes))
            real_nodes = random.sample(available_nodes, num_real) if available_nodes else []
            
            # Add fake node entries
            fake_nodes = [
                RouteNode(
                    node_id=f"decoy_{secrets.token_hex(4)}",
                    address=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                    port=random.randint(10000, 60000),
                    public_key=secrets.token_bytes(32)
                )
                for _ in range(random.randint(1, 2))
            ]
            
            path = real_nodes + fake_nodes
            random.shuffle(path)
        else:
            # Onion route - all encrypted layers
            num_hops = min(max_hops, len(available_nodes))
            path = random.sample(available_nodes, num_hops) if available_nodes else []
            
        # Filter None values
        path = [n for n in path if n is not None]
        
        route = Route(
            route_id=secrets.token_hex(8),
            route_type=route_type,
            path=path,
            created_at=time.time(),
            expires_at=time.time() + ttl_seconds
        )
        
        self.routes[route.route_id] = route
        self.stats["routes_created"] += 1
        
        return route
    
    def get_route(self, 
                  destination: str,
                  prefer_multi_hop: bool = True) -> Route:
        """
        Get best route to destination
        
        Args:
            destination: Target node
            prefer_multi_hop: Whether to prefer multi-hop routes
            
        Returns:
            Selected route
        """
        # Clean expired routes
        self._cleanup_expired_routes()
        
        # Get routes to destination
        available = [
            r for r in self.routes.values()
            if r.is_active and r.path and r.path[-1].node_id == destination
        ]
        
        # If no routes, create one
        if not available:
            route = self.create_route(destination, RouteType.MULTI_HOP)
            available = [route]
            
        return self.route_selector.select_route(available, destination, prefer_multi_hop)
    
    def route_packet(self, 
                     payload: bytes,
                     destination: str,
                     inject_decoys: bool = True) -> Tuple[Route, List[DecoyPacket]]:
        """
        Route a packet with optional decoy injection
        
        Args:
            payload: Data to route
            destination: Target node
            inject_decoys: Whether to send decoy packets
            
        Returns:
            Tuple of (used route, list of decoy packets)
        """
        # Get route
        route = self.get_route(destination)
        route.usage_count += 1
        self.stats["packets_routed"] += 1
        
        # Generate decoys
        decoys = []
        if inject_decoys:
            num_decoys = random.randint(1, 3)
            for _ in range(num_decoys):
                decoy = self.decoy_generator.generate_decoy_packet()
                decoys.append(decoy)
                self.stats["decoy_packets_sent"] += 1
                
        return route, decoys
    
    def start_decoy_traffic(self):
        """Start background decoy traffic generation"""
        destinations = list(self.peer_nodes.keys())
        self.decoy_generator.start_continuous_generation(destinations)
        
    def stop_decoy_traffic(self):
        """Stop background decoy traffic"""
        self.decoy_generator.stop_generation()
        
    def _cleanup_expired_routes(self):
        """Remove expired routes"""
        current_time = time.time()
        expired = [
            rid for rid, route in self.routes.items()
            if route.expires_at < current_time
        ]
        for rid in expired:
            del self.routes[rid]
            self.stats["routes_expired"] += 1
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            **self.stats,
            "active_routes": len([r for r in self.routes.values() if r.is_active]),
            "total_routes": len(self.routes),
            "peer_count": len(self.peer_nodes)
        }


if __name__ == "__main__":
    # Test decoy routing
    print("Testing StealthMesh Decoy Routing Module")
    print("=" * 60)
    
    # Create router
    router = DecoyRouter(
        node_id="node_1",
        randomization_factor=0.3,
        decoy_traffic_rate=0.5
    )
    
    # Register some peers
    for i in range(2, 6):
        peer = RouteNode(
            node_id=f"node_{i}",
            address=f"192.168.1.{i}",
            port=8000 + i,
            public_key=secrets.token_bytes(32),
            latency_ms=random.uniform(5, 50),
            trust_score=random.uniform(0.7, 1.0)
        )
        router.register_peer(peer)
        print(f"Registered peer: {peer.node_id} (trust: {peer.trust_score:.2f})")
        
    # Create routes
    print("\nCreating routes...")
    for rt in [RouteType.DIRECT, RouteType.MULTI_HOP, RouteType.DECOY]:
        route = router.create_route("node_5", route_type=rt)
        print(f"  {rt.value}: {len(route.path)} hops - {[n.node_id for n in route.path]}")
        
    # Test routing
    print("\nRouting packets...")
    for i in range(5):
        route, decoys = router.route_packet(b"test_payload", "node_5")
        print(f"  Packet {i+1}: Route {route.route_id[:8]}... ({route.route_type.value}), {len(decoys)} decoys")
        
    # Statistics
    print("\nStatistics:")
    stats = router.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n✓ Decoy routing test passed!")
