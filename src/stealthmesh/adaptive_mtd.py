"""
Adaptive Moving Target Defense (MTD) Module for StealthMesh
Dynamic port/service shuffling and attack surface mutation
"""

import os
import time
import secrets
import random
import threading
import socket
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class MTDStrategy(Enum):
    """Moving Target Defense strategies"""
    PORT_HOPPING = "port_hopping"
    SERVICE_SHUFFLE = "service_shuffle"
    IP_ROTATION = "ip_rotation"
    DECOY_DEPLOYMENT = "decoy_deployment"
    CONFIG_MUTATION = "config_mutation"


class MutationType(Enum):
    """Types of attack surface mutations"""
    PORT_CHANGE = "port_change"
    SERVICE_RELOCATE = "service_relocate"
    DECOY_SPAWN = "decoy_spawn"
    BANNER_CHANGE = "banner_change"
    RESPONSE_DELAY = "response_delay"


@dataclass
class ServiceConfig:
    """Configuration for a protected service"""
    service_id: str
    name: str
    base_port: int
    current_port: int
    protocol: str  # tcp, udp
    allowed_clients: Set[str] = field(default_factory=set)
    is_critical: bool = False
    last_mutation: float = 0.0
    mutation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "name": self.name,
            "base_port": self.base_port,
            "current_port": self.current_port,
            "protocol": self.protocol,
            "is_critical": self.is_critical,
            "mutation_count": self.mutation_count
        }


@dataclass
class MutationEvent:
    """Record of an attack surface mutation"""
    event_id: str
    mutation_type: MutationType
    service_id: Optional[str]
    old_value: Any
    new_value: Any
    timestamp: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "mutation_type": self.mutation_type.value,
            "service_id": self.service_id,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "timestamp": self.timestamp,
            "reason": self.reason
        }


@dataclass
class DecoyService:
    """A decoy/honeypot service"""
    decoy_id: str
    port: int
    protocol: str
    service_type: str  # ssh, http, ftp, etc.
    banner: str
    created_at: float
    interactions: int = 0
    last_interaction: float = 0.0
    attackers: Set[str] = field(default_factory=set)


class PortHopper:
    """
    Manages dynamic port hopping for services
    """
    
    def __init__(self, 
                 port_range: Tuple[int, int] = (10000, 60000),
                 reserved_ports: Set[int] = None):
        """
        Args:
            port_range: Range of ports to use for hopping
            reserved_ports: Ports to never use
        """
        self.port_range = port_range
        self.reserved_ports = reserved_ports or {22, 80, 443, 8080}
        self.used_ports: Set[int] = set()
        self.port_history: Dict[str, List[int]] = defaultdict(list)
        
    def get_new_port(self, service_id: str) -> int:
        """Get a new port for a service"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            port = random.randint(*self.port_range)
            
            # Check availability
            if port in self.reserved_ports:
                continue
            if port in self.used_ports:
                continue
                
            # Check if actually available on system
            if self._is_port_available(port):
                self.used_ports.add(port)
                self.port_history[service_id].append(port)
                return port
                
        raise RuntimeError("Could not find available port")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if port is available on the system"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                result = s.connect_ex(('127.0.0.1', port))
                return result != 0  # Port is available if connect fails
        except:
            return True
            
    def release_port(self, port: int):
        """Release a port back to the pool"""
        self.used_ports.discard(port)
        
    def get_port_history(self, service_id: str) -> List[int]:
        """Get history of ports used by a service"""
        return self.port_history.get(service_id, [])


class DecoyManager:
    """
    Manages decoy/honeypot services for deception
    """
    
    # Common service banners for deception
    DECOY_BANNERS = {
        "ssh": [
            "SSH-2.0-OpenSSH_7.4",
            "SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.1",
            "SSH-2.0-dropbear_2019.78"
        ],
        "ftp": [
            "220 (vsFTPd 3.0.3)",
            "220 ProFTPD 1.3.5 Server",
            "220 Microsoft FTP Service"
        ],
        "http": [
            "HTTP/1.1 200 OK\r\nServer: Apache/2.4.29 (Ubuntu)",
            "HTTP/1.1 200 OK\r\nServer: nginx/1.14.0",
            "HTTP/1.1 200 OK\r\nServer: Microsoft-IIS/10.0"
        ],
        "smtp": [
            "220 mail.example.com ESMTP Postfix",
            "220 smtp.example.com Microsoft ESMTP MAIL Service"
        ],
        "mysql": [
            "5.7.31-0ubuntu0.18.04.1"
        ]
    }
    
    def __init__(self, max_decoys: int = 10):
        self.max_decoys = max_decoys
        self.decoys: Dict[str, DecoyService] = {}
        self.port_hopper = PortHopper()
        
    def spawn_decoy(self, 
                    service_type: str,
                    port: int = None) -> DecoyService:
        """
        Spawn a new decoy service
        
        Args:
            service_type: Type of service to emulate
            port: Specific port (random if not specified)
            
        Returns:
            Created DecoyService
        """
        if len(self.decoys) >= self.max_decoys:
            # Remove oldest decoy
            oldest = min(self.decoys.values(), key=lambda d: d.created_at)
            self.remove_decoy(oldest.decoy_id)
            
        decoy_id = secrets.token_hex(6)
        
        if port is None:
            port = self.port_hopper.get_new_port(f"decoy_{decoy_id}")
            
        banners = self.DECOY_BANNERS.get(service_type, [""])
        banner = random.choice(banners)
        
        decoy = DecoyService(
            decoy_id=decoy_id,
            port=port,
            protocol="tcp",
            service_type=service_type,
            banner=banner,
            created_at=time.time()
        )
        
        self.decoys[decoy_id] = decoy
        return decoy
    
    def remove_decoy(self, decoy_id: str):
        """Remove a decoy service"""
        if decoy_id in self.decoys:
            decoy = self.decoys[decoy_id]
            self.port_hopper.release_port(decoy.port)
            del self.decoys[decoy_id]
            
    def record_interaction(self, decoy_id: str, attacker_ip: str):
        """Record an interaction with a decoy"""
        if decoy_id in self.decoys:
            decoy = self.decoys[decoy_id]
            decoy.interactions += 1
            decoy.last_interaction = time.time()
            decoy.attackers.add(attacker_ip)
            
    def get_suspicious_ips(self) -> Set[str]:
        """Get IPs that have interacted with decoys"""
        all_attackers = set()
        for decoy in self.decoys.values():
            all_attackers.update(decoy.attackers)
        return all_attackers
    
    def get_decoy_statistics(self) -> Dict[str, Any]:
        """Get decoy statistics"""
        total_interactions = sum(d.interactions for d in self.decoys.values())
        unique_attackers = len(self.get_suspicious_ips())
        
        return {
            "active_decoys": len(self.decoys),
            "total_interactions": total_interactions,
            "unique_attackers": unique_attackers,
            "decoy_types": list(set(d.service_type for d in self.decoys.values()))
        }


class AdaptiveMTD:
    """
    Main Adaptive Moving Target Defense engine for StealthMesh
    Coordinates all MTD strategies for attack surface mutation
    """
    
    def __init__(self,
                 mutation_interval: int = 300,  # 5 minutes default
                 enable_port_hopping: bool = True,
                 enable_decoys: bool = True,
                 dry_run: bool = True):
        """
        Initialize Adaptive MTD
        
        Args:
            mutation_interval: Seconds between automatic mutations
            enable_port_hopping: Enable port hopping strategy
            enable_decoys: Enable decoy deployment
            dry_run: Don't apply real network changes
        """
        self.mutation_interval = mutation_interval
        self.enable_port_hopping = enable_port_hopping
        self.enable_decoys = enable_decoys
        self.dry_run = dry_run
        
        # Strategy components
        self.port_hopper = PortHopper()
        self.decoy_manager = DecoyManager()
        
        # Service management
        self.services: Dict[str, ServiceConfig] = {}
        
        # Mutation tracking
        self.mutation_history: List[MutationEvent] = []
        self.last_mutation_time: float = 0.0
        
        # Threat awareness
        self.threat_level: float = 0.0  # 0-1 scale
        self.recent_attacks: List[Dict[str, Any]] = []
        
        # Callbacks
        self.mutation_callbacks: List[Callable[[MutationEvent], None]] = []
        
        # Background thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "total_mutations": 0,
            "port_hops": 0,
            "decoys_spawned": 0,
            "attack_surface_resets": 0
        }
        
    def register_callback(self, callback: Callable[[MutationEvent], None]):
        """Register callback for mutation events"""
        self.mutation_callbacks.append(callback)
        
    def register_service(self,
                          name: str,
                          port: int,
                          protocol: str = "tcp",
                          is_critical: bool = False,
                          allowed_clients: Set[str] = None) -> ServiceConfig:
        """
        Register a service for MTD protection
        
        Args:
            name: Service name
            port: Current port
            protocol: tcp or udp
            is_critical: If True, mutations are more careful
            allowed_clients: Set of allowed client IPs
            
        Returns:
            ServiceConfig
        """
        service_id = secrets.token_hex(4)
        
        service = ServiceConfig(
            service_id=service_id,
            name=name,
            base_port=port,
            current_port=port,
            protocol=protocol,
            is_critical=is_critical,
            allowed_clients=allowed_clients or set()
        )
        
        self.services[service_id] = service
        self.port_hopper.used_ports.add(port)
        
        return service
    
    def unregister_service(self, service_id: str):
        """Remove a service from MTD protection"""
        if service_id in self.services:
            service = self.services[service_id]
            self.port_hopper.release_port(service.current_port)
            del self.services[service_id]
            
    def update_threat_level(self, level: float):
        """
        Update current threat level (affects mutation frequency)
        
        Args:
            level: Threat level 0-1
        """
        self.threat_level = max(0.0, min(1.0, level))
        
    def report_attack(self, attack_info: Dict[str, Any]):
        """Report an attack for adaptive response"""
        self.recent_attacks.append({
            **attack_info,
            "timestamp": time.time()
        })
        
        # Keep only recent attacks (last hour)
        cutoff = time.time() - 3600
        self.recent_attacks = [a for a in self.recent_attacks if a["timestamp"] > cutoff]
        
        # Update threat level based on attack frequency
        attack_count = len(self.recent_attacks)
        self.threat_level = min(1.0, attack_count / 20)  # 20+ attacks = max threat
        
        # Trigger reactive mutation if threat is high
        if self.threat_level > 0.7:
            self.trigger_emergency_mutation()
            
    def mutate_service_port(self, service_id: str, reason: str = "scheduled") -> Optional[MutationEvent]:
        """
        Mutate a service to a new port
        
        Args:
            service_id: Service to mutate
            reason: Reason for mutation
            
        Returns:
            MutationEvent or None
        """
        if service_id not in self.services:
            return None
            
        service = self.services[service_id]
        old_port = service.current_port
        
        try:
            new_port = self.port_hopper.get_new_port(service_id)
        except RuntimeError:
            return None
            
        # Release old port
        self.port_hopper.release_port(old_port)
        
        # Update service
        service.current_port = new_port
        service.last_mutation = time.time()
        service.mutation_count += 1
        
        # Create mutation event
        event = MutationEvent(
            event_id=secrets.token_hex(6),
            mutation_type=MutationType.PORT_CHANGE,
            service_id=service_id,
            old_value=old_port,
            new_value=new_port,
            timestamp=time.time(),
            reason=reason
        )
        
        self.mutation_history.append(event)
        self.stats["total_mutations"] += 1
        self.stats["port_hops"] += 1
        
        # Notify callbacks
        for callback in self.mutation_callbacks:
            try:
                callback(event)
            except Exception:
                pass
                
        if not self.dry_run:
            # In real implementation, would update firewall/NAT rules
            pass
            
        print(f"[MTD] Service {service.name}: Port {old_port} -> {new_port} ({reason})")
        
        return event
    
    def spawn_decoys(self, count: int = 3, service_types: List[str] = None) -> List[DecoyService]:
        """
        Spawn multiple decoy services
        
        Args:
            count: Number of decoys to spawn
            service_types: Types of services to emulate
            
        Returns:
            List of spawned decoys
        """
        if not self.enable_decoys:
            return []
            
        if service_types is None:
            service_types = ["ssh", "ftp", "http", "mysql"]
            
        decoys = []
        for _ in range(count):
            service_type = random.choice(service_types)
            decoy = self.decoy_manager.spawn_decoy(service_type)
            decoys.append(decoy)
            self.stats["decoys_spawned"] += 1
            
            # Create mutation event
            event = MutationEvent(
                event_id=secrets.token_hex(6),
                mutation_type=MutationType.DECOY_SPAWN,
                service_id=None,
                old_value=None,
                new_value=f"{service_type}:{decoy.port}",
                timestamp=time.time(),
                reason="decoy_deployment"
            )
            self.mutation_history.append(event)
            
            print(f"[MTD] Spawned decoy: {service_type} on port {decoy.port}")
            
        return decoys
    
    def trigger_emergency_mutation(self):
        """Trigger emergency mutation of all services (high threat response)"""
        print("[MTD] Emergency mutation triggered!")
        
        for service_id in self.services:
            self.mutate_service_port(service_id, reason="emergency_high_threat")
            
        # Spawn additional decoys
        self.spawn_decoys(count=5)
        
        self.stats["attack_surface_resets"] += 1
        
    def perform_scheduled_mutation(self):
        """Perform scheduled mutation based on interval and threat level"""
        current_time = time.time()
        
        # Adjust interval based on threat level
        adjusted_interval = self.mutation_interval * (1 - self.threat_level * 0.5)
        
        if current_time - self.last_mutation_time < adjusted_interval:
            return
            
        self.last_mutation_time = current_time
        
        # Randomly select services to mutate
        services_to_mutate = [
            sid for sid, service in self.services.items()
            if not service.is_critical or random.random() < 0.3  # Critical services mutate less
        ]
        
        # Mutate subset based on threat level
        mutation_count = max(1, int(len(services_to_mutate) * (0.3 + self.threat_level * 0.4)))
        selected = random.sample(services_to_mutate, min(mutation_count, len(services_to_mutate)))
        
        for service_id in selected:
            self.mutate_service_port(service_id, reason="scheduled")
            
        # Occasionally refresh decoys
        if random.random() < 0.3:
            self.spawn_decoys(count=random.randint(1, 3))
            
    def start(self):
        """Start background mutation thread"""
        self._running = True
        self._thread = threading.Thread(target=self._mutation_loop, daemon=True)
        self._thread.start()
        print("[MTD] Adaptive MTD started")
        
    def stop(self):
        """Stop MTD engine"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[MTD] Adaptive MTD stopped")
        
    def _mutation_loop(self):
        """Background mutation loop"""
        while self._running:
            self.perform_scheduled_mutation()
            time.sleep(30)  # Check every 30 seconds
            
    def get_service_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get current port mapping for all services"""
        return {
            service.name: {
                "service_id": service.service_id,
                "base_port": service.base_port,
                "current_port": service.current_port,
                "mutations": service.mutation_count
            }
            for service in self.services.values()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MTD statistics"""
        return {
            **self.stats,
            "threat_level": self.threat_level,
            "protected_services": len(self.services),
            "recent_attacks": len(self.recent_attacks),
            **self.decoy_manager.get_decoy_statistics()
        }


if __name__ == "__main__":
    # Test Adaptive MTD
    print("Testing StealthMesh Adaptive MTD Module")
    print("=" * 60)
    
    # Create MTD engine
    mtd = AdaptiveMTD(
        mutation_interval=60,  # 1 minute for testing
        enable_port_hopping=True,
        enable_decoys=True,
        dry_run=True
    )
    
    # Register callback
    def on_mutation(event: MutationEvent):
        print(f"  [CALLBACK] Mutation: {event.mutation_type.value}")
        
    mtd.register_callback(on_mutation)
    
    # Register some services
    print("\nRegistering services...")
    web = mtd.register_service("web", port=8080, protocol="tcp")
    api = mtd.register_service("api", port=3000, protocol="tcp")
    db = mtd.register_service("database", port=5432, protocol="tcp", is_critical=True)
    
    print(f"  Registered: web (port {web.current_port})")
    print(f"  Registered: api (port {api.current_port})")
    print(f"  Registered: database (port {db.current_port}, critical)")
    
    # Test port hopping
    print("\nTesting port hopping...")
    mtd.mutate_service_port(web.service_id, reason="test")
    mtd.mutate_service_port(api.service_id, reason="test")
    
    # Show new mapping
    print("\nService Mapping:")
    for name, info in mtd.get_service_mapping().items():
        print(f"  {name}: {info['base_port']} -> {info['current_port']} (mutations: {info['mutations']})")
    
    # Test decoy deployment
    print("\nSpawning decoys...")
    decoys = mtd.spawn_decoys(count=3)
    for decoy in decoys:
        print(f"  Decoy: {decoy.service_type} on port {decoy.port}")
    
    # Simulate attacks
    print("\nSimulating attack reports...")
    for i in range(5):
        mtd.report_attack({
            "source_ip": f"10.0.0.{i}",
            "attack_type": "port_scan"
        })
    print(f"Threat level: {mtd.threat_level:.2f}")
    
    # High threat attack
    print("\nSimulating high-volume attack...")
    for i in range(15):
        mtd.report_attack({
            "source_ip": f"192.168.1.{i}",
            "attack_type": "ddos"
        })
    print(f"Threat level: {mtd.threat_level:.2f}")
    
    # Statistics
    print("\nStatistics:")
    stats = mtd.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Mutation history
    print(f"\nMutation History ({len(mtd.mutation_history)} events):")
    for event in mtd.mutation_history[-5:]:
        print(f"  {event.mutation_type.value}: {event.old_value} -> {event.new_value} ({event.reason})")
    
    print("\n✓ Adaptive MTD test passed!")
