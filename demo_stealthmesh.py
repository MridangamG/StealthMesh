"""
StealthMesh Demonstration Script
Shows all defense capabilities in action
"""

import os
import sys
import time
import random
import secrets

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stealthmesh import (
    StealthMeshNode,
    StealthCommunication,
    DecoyRouter,
    MeshCoordinator,
    ThreatDetector,
    MicroContainment,
    AdaptiveMTD
)
from src.stealthmesh.stealthmesh_node import create_stealthmesh_node, NodeConfig
from src.stealthmesh.threat_detector import NetworkFlow, ThreatLevel
from src.stealthmesh.stealth_comm import CamouflageType
from src.stealthmesh.decoy_routing import RouteNode
from src.stealthmesh.mesh_coordinator import MeshNode


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_stealth_communication():
    """Demonstrate stealth communication capabilities"""
    print_header("DEMO 1: STEALTH COMMUNICATION")
    
    print("\n[*] Creating stealth communication channel...")
    from src.stealthmesh.stealth_comm import generate_node_key
    
    key = generate_node_key()
    comm = StealthCommunication(
        master_key=key,
        camouflage_type=CamouflageType.HTTP_MIMICRY,
        rotation_interval=5
    )
    
    # Test data
    alert_data = {
        "type": "security_alert",
        "attack_type": "port_scan",
        "source_ip": "10.0.0.50",
        "severity": "high",
        "timestamp": time.time()
    }
    
    print(f"\n[*] Original message: {alert_data}")
    
    # Create stealth packet
    packet = comm.create_stealth_packet(alert_data)
    print(f"\n[*] Encrypted & Camouflaged:")
    print(f"    Cipher: {packet.cipher_type.value}")
    print(f"    Camouflage: {packet.camouflage_type.value}")
    print(f"    Payload size: {len(packet.payload)} bytes")
    print(f"    First 200 chars of payload:")
    print(f"    {packet.payload[:200].decode('utf-8', errors='ignore')}")
    
    # Serialize for network
    serialized = comm.serialize_packet(packet)
    print(f"\n[*] Serialized for transmission: {len(serialized)} bytes")
    
    # Decode
    received = comm.deserialize_packet(serialized)
    decoded = comm.decode_stealth_packet(received)
    print(f"\n[*] Decoded message: {decoded}")
    
    # Test cipher rotation
    print("\n[*] Testing cipher rotation (every 5 messages):")
    for i in range(8):
        p = comm.create_stealth_packet({"msg": i})
        print(f"    Message {i}: {p.cipher_type.value}")
    
    print("\n[✓] Stealth communication demo complete!")


def demo_decoy_routing():
    """Demonstrate decoy routing capabilities"""
    print_header("DEMO 2: DECOY ROUTING")
    
    print("\n[*] Creating decoy router...")
    router = DecoyRouter(
        node_id="router_1",
        randomization_factor=0.3,
        decoy_traffic_rate=1.0
    )
    
    # Add peers
    print("\n[*] Adding peer nodes...")
    for i in range(2, 7):
        peer = RouteNode(
            node_id=f"node_{i}",
            address=f"192.168.1.{i}",
            port=8000 + i,
            public_key=secrets.token_bytes(32),
            latency_ms=random.uniform(5, 50),
            trust_score=random.uniform(0.7, 1.0)
        )
        router.register_peer(peer)
        print(f"    Added: {peer.node_id} (trust: {peer.trust_score:.2f})")
    
    # Create routes
    print("\n[*] Creating routes to node_6:")
    from src.stealthmesh.decoy_routing import RouteType
    for rt in [RouteType.DIRECT, RouteType.MULTI_HOP, RouteType.DECOY]:
        route = router.create_route("node_6", route_type=rt)
        path_str = " -> ".join([n.node_id for n in route.path])
        print(f"    {rt.value}: {path_str}")
    
    # Route packets with decoys
    print("\n[*] Routing packets with decoy injection:")
    for i in range(5):
        route, decoys = router.route_packet(b"test_data", "node_6")
        print(f"    Packet {i+1}: Via {route.route_id[:8]}... + {len(decoys)} decoys")
    
    # Generate decoy traffic
    print("\n[*] Generating decoy traffic samples:")
    for _ in range(3):
        from src.stealthmesh.decoy_routing import TrafficType
        decoy = router.decoy_generator.generate_decoy_packet()
        print(f"    {decoy.traffic_type.value}: {len(decoy.payload)} bytes -> {decoy.destination}")
    
    print(f"\n[*] Routing statistics: {router.get_statistics()}")
    print("\n[✓] Decoy routing demo complete!")


def demo_mesh_network():
    """Demonstrate mesh network coordination"""
    print_header("DEMO 3: MESH NETWORK COORDINATION")
    
    print("\n[*] Creating mesh coordinator...")
    from src.stealthmesh.mesh_coordinator import MeshNode, Alert
    
    coordinator = MeshCoordinator(
        node_id="mesh_node_1",
        address="192.168.1.1",
        port=8001
    )
    
    # Add peers
    print("\n[*] Registering peer nodes...")
    for i in range(2, 5):
        peer = MeshNode(
            node_id=f"mesh_node_{i}",
            address=f"192.168.1.{i}",
            port=8000 + i,
            public_key=secrets.token_bytes(32)
        )
        coordinator.add_peer(peer)
        print(f"    Added peer: {peer.node_id}")
    
    # Raise alert
    print("\n[*] Raising security alert...")
    alert = coordinator.raise_alert(
        alert_type="ddos_attack",
        severity="critical",
        description="DDoS attack detected from botnet",
        target_ip="192.168.1.100",
        evidence={"pps": 1000000, "bandwidth_mbps": 500}
    )
    print(f"    Alert ID: {alert.alert_id}")
    print(f"    Type: {alert.alert_type}")
    print(f"    Severity: {alert.severity}")
    
    # Request consensus
    print("\n[*] Initiating consensus vote to quarantine suspicious node...")
    vote_id = coordinator.request_node_quarantine(
        "suspicious_node_99",
        "Multiple attack attempts detected"
    )
    print(f"    Vote ID: {vote_id}")
    
    # Simulate votes
    coordinator.consensus.cast_vote(vote_id, "mesh_node_2", True)
    coordinator.consensus.cast_vote(vote_id, "mesh_node_3", True)
    coordinator.consensus.cast_vote(vote_id, "mesh_node_4", False)
    
    status = coordinator.consensus.get_vote_status(vote_id)
    print(f"    Vote status: {status['votes_for']} for, {status['votes_against']} against")
    print(f"    Result: {'APPROVED' if status.get('result') else 'PENDING'}")
    
    print(f"\n[*] Mesh statistics: {coordinator.get_statistics()}")
    print("\n[✓] Mesh network demo complete!")


def demo_threat_detection():
    """Demonstrate ML-based threat detection"""
    print_header("DEMO 4: ML-BASED THREAT DETECTION")
    
    print("\n[*] Loading threat detector...")
    try:
        detector = ThreatDetector(model_name="xgboost")
    except:
        print("    XGBoost not available, using RandomForest...")
        detector = ThreatDetector(model_name="randomforest")
    
    print(f"    Model: {detector.model_name}")
    print(f"    Confidence threshold: {detector.confidence_threshold}")
    
    # Create test flows
    print("\n[*] Creating test network flows...")
    
    flows = [
        # Normal HTTPS traffic
        NetworkFlow(
            flow_id="normal_1",
            src_ip="192.168.1.10",
            dst_ip="8.8.8.8",
            src_port=54321,
            dst_port=443,
            protocol=6,
            timestamp=time.time(),
            flow_duration=5000,
            total_fwd_packets=50,
            total_bwd_packets=40,
            flow_bytes_per_sec=10000,
            flow_packets_per_sec=18
        ),
        # Port scan pattern
        NetworkFlow(
            flow_id="scan_1",
            src_ip="10.0.0.50",
            dst_ip="192.168.1.1",
            src_port=45678,
            dst_port=22,
            protocol=6,
            timestamp=time.time(),
            flow_duration=1,
            total_fwd_packets=1,
            total_bwd_packets=0,
            flow_bytes_per_sec=40,
            syn_flag_count=1
        ),
        # DDoS pattern
        NetworkFlow(
            flow_id="ddos_1",
            src_ip="10.0.0.100",
            dst_ip="192.168.1.100",
            src_port=12345,
            dst_port=80,
            protocol=6,
            timestamp=time.time(),
            flow_duration=100,
            total_fwd_packets=10000,
            total_bwd_packets=0,
            flow_bytes_per_sec=10000000,
            flow_packets_per_sec=100000,
            syn_flag_count=10000
        ),
    ]
    
    print("\n[*] Analyzing flows:")
    for flow in flows:
        result = detector.detect(flow)
        status = "🔴 ATTACK" if result.is_attack else "🟢 BENIGN"
        print(f"    {flow.flow_id}: {status}")
        print(f"        Confidence: {result.confidence:.2%}")
        print(f"        Threat Level: {result.threat_level.value}")
        print(f"        Inference: {detector.stats['avg_inference_time_ms']:.3f}ms")
    
    print(f"\n[*] Detection statistics: {detector.get_statistics()}")
    print("\n[✓] Threat detection demo complete!")


def demo_micro_containment():
    """Demonstrate micro-containment capabilities"""
    print_header("DEMO 5: MICRO-CONTAINMENT")
    
    print("\n[*] Creating containment engine (dry-run mode)...")
    containment = MicroContainment(dry_run=True, auto_contain=True)
    
    # Whitelist trusted IP
    print("\n[*] Adding trusted IPs to whitelist...")
    containment.policy.add_whitelist("192.168.1.1")
    containment.policy.add_whitelist("10.0.0.1")
    print("    Whitelisted: 192.168.1.1, 10.0.0.1")
    
    # Report breaches
    print("\n[*] Reporting security breaches:")
    
    breaches = [
        ("10.0.0.50", "port_scan", "Low severity - monitoring"),
        ("10.0.0.100", "brute_force", "Medium severity"),
        ("10.0.0.200", "ddos", "Critical - auto-contained"),
    ]
    
    for src_ip, attack_type, note in breaches:
        breach = containment.report_breach(
            source_ip=src_ip,
            attack_type=attack_type,
            evidence={"note": note}
        )
        print(f"    {src_ip}: {attack_type}")
        print(f"        Severity: {breach.severity.name}")
        print(f"        Contained: {breach.contained}")
    
    # Repeat offender
    print("\n[*] Simulating repeat offender (escalation)...")
    for i in range(3):
        breach = containment.report_breach(
            source_ip="10.0.0.75",
            attack_type="ssh_brute_force"
        )
    print(f"    10.0.0.75 after 3 offenses: Severity = {breach.severity.name}")
    
    # Show active containments
    print("\n[*] Active containment rules:")
    for rule in containment.get_active_containments():
        print(f"    {rule.action.value} -> {rule.target} ({rule.severity.name})")
    
    print(f"\n[*] Containment statistics: {containment.get_statistics()}")
    print("\n[✓] Micro-containment demo complete!")


def demo_adaptive_mtd():
    """Demonstrate Adaptive MTD capabilities"""
    print_header("DEMO 6: ADAPTIVE MOVING TARGET DEFENSE")
    
    print("\n[*] Creating Adaptive MTD engine...")
    mtd = AdaptiveMTD(
        mutation_interval=60,
        enable_port_hopping=True,
        enable_decoys=True,
        dry_run=True
    )
    
    # Register services
    print("\n[*] Registering services for protection:")
    services = [
        ("web_server", 80, True),
        ("api_gateway", 3000, False),
        ("database", 5432, True),
        ("cache", 6379, False),
    ]
    
    for name, port, critical in services:
        svc = mtd.register_service(name, port, is_critical=critical)
        crit_str = " [CRITICAL]" if critical else ""
        print(f"    {name}: port {port}{crit_str}")
    
    # Perform mutations
    print("\n[*] Performing port mutations:")
    for sid in list(mtd.services.keys())[:2]:
        mtd.mutate_service_port(sid, reason="demo")
    
    # Show new mapping
    print("\n[*] Current service mapping:")
    for name, info in mtd.get_service_mapping().items():
        print(f"    {name}: {info['base_port']} -> {info['current_port']}")
    
    # Spawn decoys
    print("\n[*] Deploying decoy honeypots:")
    decoys = mtd.spawn_decoys(count=4)
    for decoy in decoys:
        print(f"    {decoy.service_type} honeypot on port {decoy.port}")
    
    # Simulate attack escalation
    print("\n[*] Simulating attack escalation:")
    for i in range(10):
        mtd.report_attack({
            "source_ip": f"attacker_{i}",
            "attack_type": "probe"
        })
    print(f"    Threat level after 10 attacks: {mtd.threat_level:.2f}")
    
    for i in range(15):
        mtd.report_attack({
            "source_ip": f"botnet_{i}",
            "attack_type": "ddos"
        })
    print(f"    Threat level after 25 attacks: {mtd.threat_level:.2f}")
    
    print(f"\n[*] MTD statistics: {mtd.get_statistics()}")
    print("\n[✓] Adaptive MTD demo complete!")


def demo_full_node():
    """Demonstrate complete StealthMesh node"""
    print_header("DEMO 7: COMPLETE STEALTHMESH NODE")
    
    print("\n[*] Creating StealthMesh defense node...")
    node = create_stealthmesh_node(
        node_id="stealthmesh_demo",
        address="192.168.1.10",
        port=8001,
        containment_dry_run=True
    )
    
    # Add peers
    print("\n[*] Adding peer nodes:")
    node.add_peer("peer_alpha", "192.168.1.20", 8002)
    node.add_peer("peer_beta", "192.168.1.30", 8003)
    node.add_peer("peer_gamma", "192.168.1.40", 8004)
    
    # Register services
    print("\n[*] Registering protected services:")
    node.register_service("http_server", 80, is_critical=True)
    node.register_service("api", 3000)
    
    # Whitelist
    node.whitelist_ip("192.168.1.1")
    
    # Start node
    node.start()
    
    # Simulate traffic
    print("\n[*] Analyzing network traffic:")
    
    # Normal traffic
    for i in range(3):
        flow = NetworkFlow(
            flow_id=f"normal_{i}",
            src_ip=f"192.168.1.{100+i}",
            dst_ip="8.8.8.8",
            src_port=50000 + i,
            dst_port=443,
            protocol=6,
            timestamp=time.time(),
            flow_duration=5000,
            total_fwd_packets=50,
            total_bwd_packets=40,
            flow_bytes_per_sec=10000
        )
        result = node.analyze_flow(flow)
        if result:
            status = "ATTACK" if result.is_attack else "OK"
            print(f"    Flow {flow.flow_id}: {status}")
    
    # Attack traffic
    print("\n[*] Simulating attack traffic:")
    attack_flow = NetworkFlow(
        flow_id="attack_1",
        src_ip="10.0.0.99",
        dst_ip="192.168.1.10",
        src_port=12345,
        dst_port=22,
        protocol=6,
        timestamp=time.time(),
        flow_duration=10,
        total_fwd_packets=5000,
        total_bwd_packets=0,
        flow_bytes_per_sec=5000000,
        syn_flag_count=5000
    )
    result = node.analyze_flow(attack_flow)
    
    # Wait for async processing
    time.sleep(1)
    
    # Print status
    node.print_status()
    
    # Stop node
    node.stop()
    
    print("\n[✓] Full node demo complete!")


def main():
    """Run all demonstrations"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "    STEALTHMESH: ADAPTIVE STEALTH COMMUNICATION".center(68) + "#")
    print("#" + "    AND DECENTRALIZED DEFENSE TECHNIQUES".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\nThis demonstration showcases all StealthMesh capabilities:")
    print("  1. Stealth Communication (polymorphic encryption, camouflage)")
    print("  2. Decoy Routing (dynamic paths, fake traffic)")
    print("  3. Mesh Network (peer discovery, alert propagation)")
    print("  4. Threat Detection (ML-based classification)")
    print("  5. Micro-Containment (autonomous breach isolation)")
    print("  6. Adaptive MTD (attack surface mutation)")
    print("  7. Complete Node Integration")
    
    input("\nPress Enter to start demonstrations...")
    
    try:
        demo_stealth_communication()
        input("\nPress Enter to continue...")
        
        demo_decoy_routing()
        input("\nPress Enter to continue...")
        
        demo_mesh_network()
        input("\nPress Enter to continue...")
        
        demo_threat_detection()
        input("\nPress Enter to continue...")
        
        demo_micro_containment()
        input("\nPress Enter to continue...")
        
        demo_adaptive_mtd()
        input("\nPress Enter to continue...")
        
        demo_full_node()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("STEALTHMESH DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor more information, see the README.md file.")
    print("For research paper reference, see the project documentation.")


if __name__ == "__main__":
    main()
