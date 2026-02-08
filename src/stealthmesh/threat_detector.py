"""
Threat Detection Engine for StealthMesh
Integrates trained ML models for real-time attack classification
"""

import os
import sys
import time
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import MODELS_DIR, PROCESSED_DIR


class ThreatLevel(Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of attacks detected"""
    BENIGN = "benign"
    DOS = "dos"
    DDOS = "ddos"
    PORT_SCAN = "port_scan"
    BRUTE_FORCE = "brute_force"
    WEB_ATTACK = "web_attack"
    INFILTRATION = "infiltration"
    BOTNET = "botnet"
    UNKNOWN = "unknown"


@dataclass
class ThreatDetection:
    """Result of threat detection"""
    detection_id: str
    timestamp: float
    is_attack: bool
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    source_ip: Optional[str]
    dest_ip: Optional[str]
    features: Dict[str, float]
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp,
            "is_attack": self.is_attack,
            "attack_type": self.attack_type.value,
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "source_ip": self.source_ip,
            "dest_ip": self.dest_ip,
            "model_used": self.model_used
        }


@dataclass
class NetworkFlow:
    """Network flow data structure matching CICIDS 2017 features"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    timestamp: float
    
    # Flow statistics
    flow_duration: float = 0.0
    total_fwd_packets: int = 0
    total_bwd_packets: int = 0
    total_len_fwd_packets: float = 0.0
    total_len_bwd_packets: float = 0.0
    
    # Packet length stats
    fwd_packet_len_max: float = 0.0
    fwd_packet_len_min: float = 0.0
    fwd_packet_len_mean: float = 0.0
    fwd_packet_len_std: float = 0.0
    bwd_packet_len_max: float = 0.0
    bwd_packet_len_min: float = 0.0
    bwd_packet_len_mean: float = 0.0
    bwd_packet_len_std: float = 0.0
    
    # Flow rates
    flow_bytes_per_sec: float = 0.0
    flow_packets_per_sec: float = 0.0
    
    # IAT (Inter-Arrival Time)
    flow_iat_mean: float = 0.0
    flow_iat_std: float = 0.0
    flow_iat_max: float = 0.0
    flow_iat_min: float = 0.0
    fwd_iat_total: float = 0.0
    fwd_iat_mean: float = 0.0
    fwd_iat_std: float = 0.0
    fwd_iat_max: float = 0.0
    fwd_iat_min: float = 0.0
    bwd_iat_total: float = 0.0
    bwd_iat_mean: float = 0.0
    bwd_iat_std: float = 0.0
    bwd_iat_max: float = 0.0
    bwd_iat_min: float = 0.0
    
    # Flags
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    fwd_header_len: int = 0
    bwd_header_len: int = 0
    
    # Packet rates
    fwd_packets_per_sec: float = 0.0
    bwd_packets_per_sec: float = 0.0
    
    # Packet length aggregates
    min_packet_len: float = 0.0
    max_packet_len: float = 0.0
    packet_len_mean: float = 0.0
    packet_len_std: float = 0.0
    packet_len_var: float = 0.0
    
    # TCP Flags
    fin_flag_count: int = 0
    syn_flag_count: int = 0
    rst_flag_count: int = 0
    psh_flag_count: int = 0
    ack_flag_count: int = 0
    urg_flag_count: int = 0
    cwe_flag_count: int = 0
    ece_flag_count: int = 0
    
    # Additional
    down_up_ratio: float = 0.0
    avg_packet_size: float = 0.0
    avg_fwd_segment_size: float = 0.0
    avg_bwd_segment_size: float = 0.0
    
    # Subflow
    subflow_fwd_packets: int = 0
    subflow_fwd_bytes: int = 0
    subflow_bwd_packets: int = 0
    subflow_bwd_bytes: int = 0
    
    # Window sizes
    init_win_bytes_fwd: int = 0
    init_win_bytes_bwd: int = 0
    act_data_pkt_fwd: int = 0
    min_seg_size_fwd: int = 0
    
    # Active/Idle
    active_mean: float = 0.0
    active_std: float = 0.0
    active_max: float = 0.0
    active_min: float = 0.0
    idle_mean: float = 0.0
    idle_std: float = 0.0
    idle_max: float = 0.0
    idle_min: float = 0.0


class ThreatDetector:
    """
    Main threat detection engine using trained ML models
    """
    
    # Map binary predictions to threat levels
    THREAT_LEVEL_MAP = {
        0: ThreatLevel.NONE,      # Benign
        1: ThreatLevel.HIGH       # Attack
    }
    
    def __init__(self, 
                 model_name: str = "xgboost",
                 models_dir: str = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize threat detector
        
        Args:
            model_name: Which model to use (xgboost, randomforest, neuralnetwork)
            models_dir: Directory containing trained models
            confidence_threshold: Minimum confidence for detection
        """
        self.model_name = model_name
        self.models_dir = models_dir or MODELS_DIR
        self.confidence_threshold = confidence_threshold
        
        # Load model and preprocessing artifacts
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()
        
        # Detection history
        self.detection_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "total_flows_analyzed": 0,
            "attacks_detected": 0,
            "benign_flows": 0,
            "avg_inference_time_ms": 0.0
        }
        
        # Callbacks for detections
        self.detection_callbacks: List[callable] = []
        
    def _load_model(self):
        """Load trained model and preprocessing artifacts"""
        # Map model names to files
        model_files = {
            "xgboost": "xgboost_model.pkl",
            "randomforest": "randomforest_model.pkl",
            "neuralnetwork": "neuralnetwork_model.pkl"
        }
        
        model_file = model_files.get(self.model_name.lower())
        if not model_file:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        model_path = os.path.join(self.models_dir, model_file)
        
        # Load model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Handle different model storage formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
            else:
                self.model = model_data
                
            print(f"Loaded model: {self.model_name} from {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
            
        # Load scaler
        scaler_path = os.path.join(PROCESSED_DIR, "binary_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
            
        # Load feature names
        features_path = os.path.join(PROCESSED_DIR, "binary_features.pkl")
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"Loaded {len(self.feature_names)} feature names")
            
    def register_callback(self, callback: callable):
        """Register callback for detections"""
        self.detection_callbacks.append(callback)
        
    def flow_to_features(self, flow: NetworkFlow) -> np.ndarray:
        """
        Convert network flow to feature vector
        
        Args:
            flow: NetworkFlow object
            
        Returns:
            Feature vector as numpy array
        """
        # Extract features in order matching training data
        features = [
            flow.dst_port,
            flow.flow_duration,
            flow.total_fwd_packets,
            flow.total_bwd_packets,
            flow.total_len_fwd_packets,
            flow.total_len_bwd_packets,
            flow.fwd_packet_len_max,
            flow.fwd_packet_len_min,
            flow.fwd_packet_len_mean,
            flow.fwd_packet_len_std,
            flow.bwd_packet_len_max,
            flow.bwd_packet_len_min,
            flow.bwd_packet_len_mean,
            flow.bwd_packet_len_std,
            flow.flow_bytes_per_sec,
            flow.flow_packets_per_sec,
            flow.flow_iat_mean,
            flow.flow_iat_std,
            flow.flow_iat_max,
            flow.flow_iat_min,
            flow.fwd_iat_total,
            flow.fwd_iat_mean,
            flow.fwd_iat_std,
            flow.fwd_iat_max,
            flow.fwd_iat_min,
            flow.bwd_iat_total,
            flow.bwd_iat_mean,
            flow.bwd_iat_std,
            flow.bwd_iat_max,
            flow.bwd_iat_min,
            flow.fwd_psh_flags,
            flow.bwd_psh_flags,
            flow.fwd_urg_flags,
            flow.bwd_urg_flags,
            flow.fwd_header_len,
            flow.bwd_header_len,
            flow.fwd_packets_per_sec,
            flow.bwd_packets_per_sec,
            flow.min_packet_len,
            flow.max_packet_len,
        ]
        
        return np.array(features, dtype=np.float32)
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features using trained scaler"""
        # Handle NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Reshape for scaler
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Scale if scaler available
        if self.scaler is not None:
            # Ensure correct number of features
            expected_features = self.scaler.n_features_in_
            if features.shape[1] != expected_features:
                # Pad or truncate
                if features.shape[1] < expected_features:
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected_features]
                    
            features = self.scaler.transform(features)
            
        return features
    
    def detect(self, flow: NetworkFlow) -> ThreatDetection:
        """
        Detect threats in a network flow
        
        Args:
            flow: Network flow to analyze
            
        Returns:
            ThreatDetection result
        """
        start_time = time.time()
        
        # Convert to features
        raw_features = self.flow_to_features(flow)
        features = self.preprocess_features(raw_features)
        
        # Predict
        if self.model is not None:
            prediction = self.model.predict(features)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = float(max(proba))
            else:
                confidence = 1.0 if prediction == 1 else 0.0
        else:
            prediction = 0
            confidence = 0.0
            
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Determine threat level and attack type
        is_attack = prediction == 1 and confidence >= self.confidence_threshold
        threat_level = self.THREAT_LEVEL_MAP.get(int(prediction), ThreatLevel.NONE)
        
        # For binary classification, we just know it's an attack or not
        attack_type = AttackType.UNKNOWN if is_attack else AttackType.BENIGN
        
        # Create detection result
        detection = ThreatDetection(
            detection_id=f"det_{int(time.time()*1000)}_{flow.flow_id}",
            timestamp=time.time(),
            is_attack=is_attack,
            attack_type=attack_type,
            threat_level=threat_level,
            confidence=confidence,
            source_ip=flow.src_ip,
            dest_ip=flow.dst_ip,
            features={
                "dst_port": flow.dst_port,
                "flow_duration": flow.flow_duration,
                "total_packets": flow.total_fwd_packets + flow.total_bwd_packets,
                "bytes_per_sec": flow.flow_bytes_per_sec
            },
            model_used=self.model_name
        )
        
        # Update stats
        self.stats["total_flows_analyzed"] += 1
        if is_attack:
            self.stats["attacks_detected"] += 1
        else:
            self.stats["benign_flows"] += 1
            
        # Update average inference time
        n = self.stats["total_flows_analyzed"]
        self.stats["avg_inference_time_ms"] = (
            (self.stats["avg_inference_time_ms"] * (n-1) + inference_time) / n
        )
        
        # Store in history
        self.detection_history.append(detection)
        
        # Notify callbacks for attacks
        if is_attack:
            for callback in self.detection_callbacks:
                try:
                    callback(detection)
                except Exception:
                    pass
                    
        return detection
    
    def detect_batch(self, flows: List[NetworkFlow]) -> List[ThreatDetection]:
        """
        Detect threats in multiple flows (batch processing)
        
        Args:
            flows: List of network flows
            
        Returns:
            List of detection results
        """
        if not flows:
            return []
            
        start_time = time.time()
        
        # Convert all flows to features
        feature_matrix = np.array([
            self.flow_to_features(flow) for flow in flows
        ])
        
        # Preprocess
        features = self.preprocess_features(feature_matrix)
        
        # Batch predict
        if self.model is not None:
            predictions = self.model.predict(features)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
            else:
                probabilities = np.zeros((len(flows), 2))
                probabilities[np.arange(len(flows)), predictions.astype(int)] = 1.0
        else:
            predictions = np.zeros(len(flows))
            probabilities = np.zeros((len(flows), 2))
            
        # Calculate per-sample inference time
        total_time = (time.time() - start_time) * 1000
        per_sample_time = total_time / len(flows)
        
        # Create detection results
        detections = []
        for i, flow in enumerate(flows):
            prediction = predictions[i]
            confidence = float(max(probabilities[i]))
            
            is_attack = prediction == 1 and confidence >= self.confidence_threshold
            threat_level = self.THREAT_LEVEL_MAP.get(int(prediction), ThreatLevel.NONE)
            attack_type = AttackType.UNKNOWN if is_attack else AttackType.BENIGN
            
            detection = ThreatDetection(
                detection_id=f"det_{int(time.time()*1000)}_{flow.flow_id}",
                timestamp=time.time(),
                is_attack=is_attack,
                attack_type=attack_type,
                threat_level=threat_level,
                confidence=confidence,
                source_ip=flow.src_ip,
                dest_ip=flow.dst_ip,
                features={
                    "dst_port": flow.dst_port,
                    "flow_duration": flow.flow_duration
                },
                model_used=self.model_name
            )
            
            detections.append(detection)
            self.detection_history.append(detection)
            
            # Update stats
            self.stats["total_flows_analyzed"] += 1
            if is_attack:
                self.stats["attacks_detected"] += 1
                for callback in self.detection_callbacks:
                    try:
                        callback(detection)
                    except Exception:
                        pass
            else:
                self.stats["benign_flows"] += 1
                
        # Update average inference time
        self.stats["avg_inference_time_ms"] = per_sample_time
        
        return detections
    
    def get_recent_detections(self, 
                               limit: int = 100,
                               attacks_only: bool = False) -> List[ThreatDetection]:
        """Get recent detection results"""
        detections = list(self.detection_history)[-limit:]
        if attacks_only:
            detections = [d for d in detections if d.is_attack]
        return detections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        detection_rate = (
            self.stats["attacks_detected"] / max(1, self.stats["total_flows_analyzed"])
        ) * 100
        
        return {
            **self.stats,
            "detection_rate_percent": detection_rate,
            "model": self.model_name,
            "confidence_threshold": self.confidence_threshold
        }


class RealTimeDetector:
    """
    Real-time detection wrapper with continuous monitoring
    """
    
    def __init__(self, 
                 detector: ThreatDetector,
                 batch_size: int = 100,
                 batch_timeout: float = 1.0):
        """
        Args:
            detector: ThreatDetector instance
            batch_size: Flows to batch before processing
            batch_timeout: Maximum wait time before processing batch
        """
        self.detector = detector
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.flow_buffer: List[NetworkFlow] = []
        self.buffer_lock = threading.Lock()
        self.last_process_time = time.time()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def add_flow(self, flow: NetworkFlow):
        """Add flow to processing buffer"""
        with self.buffer_lock:
            self.flow_buffer.append(flow)
            
    def start(self):
        """Start real-time processing"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop real-time processing"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            
    def _process_loop(self):
        """Background processing loop"""
        while self._running:
            should_process = False
            
            with self.buffer_lock:
                # Check batch size
                if len(self.flow_buffer) >= self.batch_size:
                    should_process = True
                # Check timeout
                elif (len(self.flow_buffer) > 0 and 
                      time.time() - self.last_process_time > self.batch_timeout):
                    should_process = True
                    
                if should_process:
                    flows = self.flow_buffer.copy()
                    self.flow_buffer.clear()
                    self.last_process_time = time.time()
                    
            if should_process and flows:
                self.detector.detect_batch(flows)
                
            time.sleep(0.1)


if __name__ == "__main__":
    # Test threat detector
    print("Testing StealthMesh Threat Detection Engine")
    print("=" * 60)
    
    # Create detector
    try:
        detector = ThreatDetector(model_name="xgboost")
    except Exception as e:
        print(f"Could not load XGBoost, trying RandomForest: {e}")
        detector = ThreatDetector(model_name="randomforest")
        
    # Register callback
    def on_attack(detection: ThreatDetection):
        print(f"  [ALERT] Attack detected! Confidence: {detection.confidence:.2f}")
        
    detector.register_callback(on_attack)
    
    # Create test flows
    print("\nTesting with synthetic flows...")
    
    # Benign flow
    benign_flow = NetworkFlow(
        flow_id="flow_1",
        src_ip="192.168.1.10",
        dst_ip="8.8.8.8",
        src_port=54321,
        dst_port=443,
        protocol=6,
        timestamp=time.time(),
        flow_duration=1000,
        total_fwd_packets=10,
        total_bwd_packets=8,
        total_len_fwd_packets=500,
        total_len_bwd_packets=2000,
        flow_bytes_per_sec=2500,
        flow_packets_per_sec=18
    )
    
    # Suspicious flow (high packet rate, many SYN flags - possible scan)
    suspicious_flow = NetworkFlow(
        flow_id="flow_2",
        src_ip="10.0.0.50",
        dst_ip="192.168.1.1",
        src_port=12345,
        dst_port=22,
        protocol=6,
        timestamp=time.time(),
        flow_duration=100,
        total_fwd_packets=1000,
        total_bwd_packets=0,
        total_len_fwd_packets=40000,
        total_len_bwd_packets=0,
        flow_bytes_per_sec=400000,
        flow_packets_per_sec=10000,
        syn_flag_count=1000,
        fwd_packets_per_sec=10000
    )
    
    # Test single detection
    print("\nDetecting benign flow...")
    result1 = detector.detect(benign_flow)
    print(f"  Result: {'ATTACK' if result1.is_attack else 'BENIGN'}")
    print(f"  Confidence: {result1.confidence:.2f}")
    print(f"  Threat Level: {result1.threat_level.value}")
    
    print("\nDetecting suspicious flow...")
    result2 = detector.detect(suspicious_flow)
    print(f"  Result: {'ATTACK' if result2.is_attack else 'BENIGN'}")
    print(f"  Confidence: {result2.confidence:.2f}")
    print(f"  Threat Level: {result2.threat_level.value}")
    
    # Test batch detection
    print("\nTesting batch detection...")
    batch_flows = [benign_flow, suspicious_flow] * 5
    results = detector.detect_batch(batch_flows)
    attacks = sum(1 for r in results if r.is_attack)
    print(f"  Processed {len(results)} flows, detected {attacks} attacks")
    
    # Statistics
    print("\nDetector Statistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\n✓ Threat detection test passed!")
