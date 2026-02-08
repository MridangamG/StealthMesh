"""
Micro-Containment Engine for StealthMesh
Autonomous breach isolation and quarantine mechanisms
"""

import os
import time
import secrets
import threading
import subprocess
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ContainmentAction(Enum):
    """Types of containment actions"""
    BLOCK_IP = "block_ip"
    BLOCK_PORT = "block_port"
    RATE_LIMIT = "rate_limit"
    QUARANTINE_HOST = "quarantine_host"
    ISOLATE_SEGMENT = "isolate_segment"
    REDIRECT_TO_HONEYPOT = "redirect_honeypot"
    DROP_CONNECTION = "drop_connection"


class ContainmentStatus(Enum):
    """Status of containment action"""
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    FAILED = "failed"


class ThreatSeverity(Enum):
    """Threat severity for containment decisions"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContainmentRule:
    """A containment rule/action"""
    rule_id: str
    action: ContainmentAction
    target: str  # IP, port, or host
    severity: ThreatSeverity
    reason: str
    created_at: float
    expires_at: float
    status: ContainmentStatus = ContainmentStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": self.action.value,
            "target": self.target,
            "severity": self.severity.name,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "metadata": self.metadata
        }


@dataclass  
class Breach:
    """Detected breach information"""
    breach_id: str
    source_ip: str
    target_ip: Optional[str]
    attack_type: str
    severity: ThreatSeverity
    timestamp: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    contained: bool = False
    containment_rules: List[str] = field(default_factory=list)


class ContainmentPolicy:
    """
    Policy engine for automatic containment decisions
    """
    
    # Default policies by severity
    DEFAULT_POLICIES = {
        ThreatSeverity.LOW: {
            "action": ContainmentAction.RATE_LIMIT,
            "duration_seconds": 300,  # 5 minutes
            "auto_apply": False
        },
        ThreatSeverity.MEDIUM: {
            "action": ContainmentAction.BLOCK_PORT,
            "duration_seconds": 1800,  # 30 minutes
            "auto_apply": True
        },
        ThreatSeverity.HIGH: {
            "action": ContainmentAction.BLOCK_IP,
            "duration_seconds": 3600,  # 1 hour
            "auto_apply": True
        },
        ThreatSeverity.CRITICAL: {
            "action": ContainmentAction.QUARANTINE_HOST,
            "duration_seconds": 86400,  # 24 hours
            "auto_apply": True
        }
    }
    
    # Attack type to severity mapping
    ATTACK_SEVERITY = {
        "port_scan": ThreatSeverity.LOW,
        "brute_force": ThreatSeverity.MEDIUM,
        "dos": ThreatSeverity.HIGH,
        "ddos": ThreatSeverity.CRITICAL,
        "web_attack": ThreatSeverity.MEDIUM,
        "infiltration": ThreatSeverity.HIGH,
        "botnet": ThreatSeverity.CRITICAL,
        "malware": ThreatSeverity.CRITICAL,
        "unknown": ThreatSeverity.MEDIUM
    }
    
    def __init__(self):
        self.custom_policies: Dict[str, Dict[str, Any]] = {}
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        
    def add_whitelist(self, ip: str):
        """Add IP to whitelist (never contain)"""
        self.whitelist.add(ip)
        self.blacklist.discard(ip)
        
    def add_blacklist(self, ip: str):
        """Add IP to permanent blacklist"""
        self.blacklist.add(ip)
        self.whitelist.discard(ip)
        
    def get_severity(self, attack_type: str) -> ThreatSeverity:
        """Get severity for attack type"""
        return self.ATTACK_SEVERITY.get(
            attack_type.lower(), 
            ThreatSeverity.MEDIUM
        )
        
    def should_auto_contain(self, 
                            source_ip: str, 
                            severity: ThreatSeverity) -> bool:
        """Check if automatic containment should be applied"""
        # Never contain whitelisted IPs
        if source_ip in self.whitelist:
            return False
            
        # Always contain blacklisted IPs
        if source_ip in self.blacklist:
            return True
            
        policy = self.DEFAULT_POLICIES.get(severity, {})
        return policy.get("auto_apply", False)
    
    def get_containment_action(self, 
                                severity: ThreatSeverity,
                                attack_type: str = None) -> tuple:
        """
        Get recommended containment action for severity
        
        Returns:
            Tuple of (action, duration_seconds)
        """
        policy = self.DEFAULT_POLICIES.get(severity, self.DEFAULT_POLICIES[ThreatSeverity.MEDIUM])
        
        # Check for attack-specific custom policy
        if attack_type and attack_type in self.custom_policies:
            policy = self.custom_policies[attack_type]
            
        return (policy["action"], policy["duration_seconds"])


class FirewallInterface:
    """
    Interface to system firewall for applying containment rules
    Supports iptables (Linux) and Windows Firewall
    """
    
    def __init__(self, dry_run: bool = True):
        """
        Args:
            dry_run: If True, don't actually apply rules (for testing)
        """
        self.dry_run = dry_run
        self.platform = self._detect_platform()
        self.applied_rules: Dict[str, str] = {}  # rule_id -> command
        
    def _detect_platform(self) -> str:
        """Detect operating system"""
        import platform
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        else:
            return "unknown"
            
    def apply_rule(self, rule: ContainmentRule) -> bool:
        """
        Apply containment rule to firewall
        
        Args:
            rule: ContainmentRule to apply
            
        Returns:
            Success status
        """
        if self.dry_run:
            print(f"[DRY RUN] Would apply rule: {rule.action.value} on {rule.target}")
            self.applied_rules[rule.rule_id] = f"dry_run_{rule.action.value}"
            return True
            
        try:
            if self.platform == "linux":
                return self._apply_linux(rule)
            elif self.platform == "windows":
                return self._apply_windows(rule)
            else:
                print(f"Unsupported platform: {self.platform}")
                return False
        except Exception as e:
            print(f"Error applying rule: {e}")
            return False
            
    def _apply_linux(self, rule: ContainmentRule) -> bool:
        """Apply rule using iptables"""
        commands = []
        
        if rule.action == ContainmentAction.BLOCK_IP:
            commands = [
                f"iptables -A INPUT -s {rule.target} -j DROP",
                f"iptables -A OUTPUT -d {rule.target} -j DROP"
            ]
        elif rule.action == ContainmentAction.BLOCK_PORT:
            port = rule.metadata.get("port", rule.target)
            commands = [
                f"iptables -A INPUT -p tcp --dport {port} -j DROP",
                f"iptables -A INPUT -p udp --dport {port} -j DROP"
            ]
        elif rule.action == ContainmentAction.RATE_LIMIT:
            limit = rule.metadata.get("limit", "10/minute")
            commands = [
                f"iptables -A INPUT -s {rule.target} -m limit --limit {limit} -j ACCEPT",
                f"iptables -A INPUT -s {rule.target} -j DROP"
            ]
        elif rule.action == ContainmentAction.DROP_CONNECTION:
            commands = [
                f"iptables -A INPUT -s {rule.target} -j REJECT --reject-with tcp-reset"
            ]
            
        for cmd in commands:
            result = subprocess.run(cmd.split(), capture_output=True)
            if result.returncode != 0:
                return False
                
        self.applied_rules[rule.rule_id] = "; ".join(commands)
        return True
        
    def _apply_windows(self, rule: ContainmentRule) -> bool:
        """Apply rule using Windows Firewall (netsh)"""
        rule_name = f"StealthMesh_{rule.rule_id[:8]}"
        
        if rule.action == ContainmentAction.BLOCK_IP:
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={rule.target}'
        elif rule.action == ContainmentAction.BLOCK_PORT:
            port = rule.metadata.get("port", rule.target)
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block localport={port} protocol=tcp'
        else:
            # Windows has limited options
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={rule.target}'
            
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            self.applied_rules[rule.rule_id] = cmd
            return True
        return False
        
    def remove_rule(self, rule: ContainmentRule) -> bool:
        """Remove a previously applied rule"""
        if self.dry_run:
            print(f"[DRY RUN] Would remove rule: {rule.rule_id}")
            self.applied_rules.pop(rule.rule_id, None)
            return True
            
        try:
            if self.platform == "linux":
                return self._remove_linux(rule)
            elif self.platform == "windows":
                return self._remove_windows(rule)
            return False
        except Exception as e:
            print(f"Error removing rule: {e}")
            return False
            
    def _remove_linux(self, rule: ContainmentRule) -> bool:
        """Remove iptables rule"""
        # Replace -A with -D in original commands
        original = self.applied_rules.get(rule.rule_id, "")
        if original:
            commands = original.replace("-A", "-D").split("; ")
            for cmd in commands:
                subprocess.run(cmd.split(), capture_output=True)
        self.applied_rules.pop(rule.rule_id, None)
        return True
        
    def _remove_windows(self, rule: ContainmentRule) -> bool:
        """Remove Windows Firewall rule"""
        rule_name = f"StealthMesh_{rule.rule_id[:8]}"
        cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
        subprocess.run(cmd, shell=True, capture_output=True)
        self.applied_rules.pop(rule.rule_id, None)
        return True


class MicroContainment:
    """
    Main micro-containment engine for StealthMesh
    Handles autonomous breach isolation and quarantine
    """
    
    def __init__(self, 
                 dry_run: bool = True,
                 auto_contain: bool = True):
        """
        Initialize micro-containment engine
        
        Args:
            dry_run: If True, don't apply real firewall rules
            auto_contain: Enable automatic containment
        """
        self.dry_run = dry_run
        self.auto_contain = auto_contain
        
        # Policy and firewall
        self.policy = ContainmentPolicy()
        self.firewall = FirewallInterface(dry_run=dry_run)
        
        # Breach and rule tracking
        self.breaches: Dict[str, Breach] = {}
        self.active_rules: Dict[str, ContainmentRule] = {}
        self.rule_history: List[ContainmentRule] = []
        
        # IP reputation tracking
        self.ip_offenses: Dict[str, int] = defaultdict(int)
        self.ip_last_offense: Dict[str, float] = {}
        
        # Callbacks
        self.containment_callbacks: List[Callable[[ContainmentRule], None]] = []
        
        # Background thread for rule expiration
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "breaches_detected": 0,
            "rules_created": 0,
            "rules_applied": 0,
            "rules_expired": 0,
            "auto_containments": 0
        }
        
    def register_callback(self, callback: Callable[[ContainmentRule], None]):
        """Register callback for containment actions"""
        self.containment_callbacks.append(callback)
        
    def report_breach(self,
                      source_ip: str,
                      attack_type: str,
                      target_ip: str = None,
                      evidence: Dict[str, Any] = None,
                      severity: ThreatSeverity = None) -> Breach:
        """
        Report a security breach for potential containment
        
        Args:
            source_ip: Source IP of the attack
            attack_type: Type of attack
            target_ip: Target IP if known
            evidence: Supporting evidence
            severity: Override severity
            
        Returns:
            Created Breach object
        """
        # Determine severity
        if severity is None:
            severity = self.policy.get_severity(attack_type)
            
        # Track offenses
        self.ip_offenses[source_ip] += 1
        self.ip_last_offense[source_ip] = time.time()
        
        # Escalate severity for repeat offenders
        if self.ip_offenses[source_ip] >= 3:
            severity = ThreatSeverity(min(severity.value + 1, ThreatSeverity.CRITICAL.value))
            
        # Create breach record
        breach = Breach(
            breach_id=secrets.token_hex(8),
            source_ip=source_ip,
            target_ip=target_ip,
            attack_type=attack_type,
            severity=severity,
            timestamp=time.time(),
            evidence=evidence or {}
        )
        
        self.breaches[breach.breach_id] = breach
        self.stats["breaches_detected"] += 1
        
        # Auto-contain if enabled
        if self.auto_contain and self.policy.should_auto_contain(source_ip, severity):
            self._auto_contain_breach(breach)
            
        return breach
    
    def _auto_contain_breach(self, breach: Breach):
        """Automatically contain a breach based on policy"""
        action, duration = self.policy.get_containment_action(
            breach.severity, 
            breach.attack_type
        )
        
        rule = self.create_containment_rule(
            action=action,
            target=breach.source_ip,
            severity=breach.severity,
            reason=f"Auto-containment: {breach.attack_type}",
            duration_seconds=duration,
            metadata={
                "breach_id": breach.breach_id,
                "attack_type": breach.attack_type,
                "auto_applied": True
            }
        )
        
        if rule:
            breach.contained = True
            breach.containment_rules.append(rule.rule_id)
            self.stats["auto_containments"] += 1
            
    def create_containment_rule(self,
                                 action: ContainmentAction,
                                 target: str,
                                 severity: ThreatSeverity,
                                 reason: str,
                                 duration_seconds: int = 3600,
                                 metadata: Dict[str, Any] = None) -> Optional[ContainmentRule]:
        """
        Create and apply a containment rule
        
        Args:
            action: Type of containment action
            target: Target (IP, port, etc.)
            severity: Threat severity
            reason: Reason for containment
            duration_seconds: How long to maintain containment
            metadata: Additional metadata
            
        Returns:
            Created rule or None if failed
        """
        # Check whitelist
        if target in self.policy.whitelist:
            print(f"Target {target} is whitelisted, skipping containment")
            return None
            
        rule = ContainmentRule(
            rule_id=secrets.token_hex(8),
            action=action,
            target=target,
            severity=severity,
            reason=reason,
            created_at=time.time(),
            expires_at=time.time() + duration_seconds,
            metadata=metadata or {}
        )
        
        self.stats["rules_created"] += 1
        
        # Apply rule
        if self.firewall.apply_rule(rule):
            rule.status = ContainmentStatus.ACTIVE
            self.active_rules[rule.rule_id] = rule
            self.rule_history.append(rule)
            self.stats["rules_applied"] += 1
            
            # Notify callbacks
            for callback in self.containment_callbacks:
                try:
                    callback(rule)
                except Exception:
                    pass
                    
            print(f"Containment applied: {action.value} on {target} for {duration_seconds}s")
            return rule
        else:
            rule.status = ContainmentStatus.FAILED
            return None
            
    def release_containment(self, rule_id: str) -> bool:
        """
        Manually release a containment rule
        
        Args:
            rule_id: ID of rule to release
            
        Returns:
            Success status
        """
        if rule_id not in self.active_rules:
            return False
            
        rule = self.active_rules[rule_id]
        
        if self.firewall.remove_rule(rule):
            rule.status = ContainmentStatus.RELEASED
            del self.active_rules[rule_id]
            print(f"Containment released: {rule.action.value} on {rule.target}")
            return True
            
        return False
    
    def get_active_containments(self) -> List[ContainmentRule]:
        """Get all active containment rules"""
        return list(self.active_rules.values())
    
    def get_breach_status(self, breach_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific breach"""
        if breach_id not in self.breaches:
            return None
            
        breach = self.breaches[breach_id]
        return {
            "breach_id": breach.breach_id,
            "source_ip": breach.source_ip,
            "attack_type": breach.attack_type,
            "severity": breach.severity.name,
            "contained": breach.contained,
            "containment_rules": [
                self.active_rules.get(rid, {})
                for rid in breach.containment_rules
            ]
        }
    
    def start(self):
        """Start background cleanup thread"""
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
        
    def stop(self):
        """Stop background thread and release all containments"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
            
        # Release all active rules
        for rule_id in list(self.active_rules.keys()):
            self.release_containment(rule_id)
            
    def _cleanup_loop(self):
        """Background loop to expire old rules"""
        while self._running:
            current_time = time.time()
            expired = []
            
            for rule_id, rule in list(self.active_rules.items()):
                if rule.expires_at < current_time:
                    expired.append(rule_id)
                    
            for rule_id in expired:
                rule = self.active_rules[rule_id]
                self.firewall.remove_rule(rule)
                rule.status = ContainmentStatus.EXPIRED
                del self.active_rules[rule_id]
                self.stats["rules_expired"] += 1
                print(f"Containment expired: {rule.action.value} on {rule.target}")
                
            # Also decay IP offense counts
            for ip in list(self.ip_offenses.keys()):
                last_offense = self.ip_last_offense.get(ip, 0)
                if current_time - last_offense > 3600:  # 1 hour decay
                    self.ip_offenses[ip] = max(0, self.ip_offenses[ip] - 1)
                    if self.ip_offenses[ip] == 0:
                        del self.ip_offenses[ip]
                        self.ip_last_offense.pop(ip, None)
                        
            time.sleep(10)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get containment statistics"""
        return {
            **self.stats,
            "active_rules": len(self.active_rules),
            "tracked_ips": len(self.ip_offenses),
            "whitelisted_ips": len(self.policy.whitelist),
            "blacklisted_ips": len(self.policy.blacklist)
        }


if __name__ == "__main__":
    # Test micro-containment
    print("Testing StealthMesh Micro-Containment Engine")
    print("=" * 60)
    
    # Create containment engine (dry run mode)
    containment = MicroContainment(dry_run=True, auto_contain=True)
    
    # Register callback
    def on_containment(rule: ContainmentRule):
        print(f"  [CALLBACK] Containment applied: {rule.action.value}")
        
    containment.register_callback(on_containment)
    
    # Add whitelist
    containment.policy.add_whitelist("192.168.1.1")
    print("Added 192.168.1.1 to whitelist")
    
    # Report some breaches
    print("\nReporting breaches...")
    
    # Low severity
    breach1 = containment.report_breach(
        source_ip="10.0.0.50",
        attack_type="port_scan",
        evidence={"ports_scanned": [22, 80, 443]}
    )
    print(f"Breach 1: {breach1.attack_type} from {breach1.source_ip} (severity: {breach1.severity.name})")
    
    # High severity
    breach2 = containment.report_breach(
        source_ip="10.0.0.100",
        attack_type="dos",
        evidence={"packets_per_sec": 100000}
    )
    print(f"Breach 2: {breach2.attack_type} from {breach2.source_ip} (severity: {breach2.severity.name})")
    
    # Critical - should be auto-contained
    breach3 = containment.report_breach(
        source_ip="10.0.0.200",
        attack_type="ddos",
        evidence={"botnet_size": 1000}
    )
    print(f"Breach 3: {breach3.attack_type} from {breach3.source_ip} (severity: {breach3.severity.name})")
    
    # Repeat offender - should escalate
    for i in range(3):
        breach4 = containment.report_breach(
            source_ip="10.0.0.75",
            attack_type="brute_force"
        )
    print(f"Repeat offender breach: {breach4.attack_type} (severity: {breach4.severity.name})")
    
    # Whitelisted IP - should not be contained
    breach5 = containment.report_breach(
        source_ip="192.168.1.1",
        attack_type="ddos"
    )
    print(f"Whitelisted breach: {breach5.contained}")
    
    # Show active containments
    print("\nActive Containments:")
    for rule in containment.get_active_containments():
        print(f"  - {rule.action.value} on {rule.target} ({rule.severity.name})")
        
    # Manual containment
    print("\nApplying manual containment...")
    manual_rule = containment.create_containment_rule(
        action=ContainmentAction.BLOCK_PORT,
        target="8080",
        severity=ThreatSeverity.MEDIUM,
        reason="Suspicious service on port 8080",
        duration_seconds=600,
        metadata={"port": 8080}
    )
    
    # Statistics
    print("\nStatistics:")
    stats = containment.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n✓ Micro-containment test passed!")
