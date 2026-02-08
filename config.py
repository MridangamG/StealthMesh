"""
StealthMesh Configuration File
Contains all project settings, paths, and hyperparameters
"""

import os

# ============== PATH CONFIGURATION ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "CIC-IDS-2017 Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============== DATASET FILES ==============
DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# ============== LABEL MAPPING ==============
# Binary classification: 0 = Benign, 1 = Attack
BINARY_LABELS = {
    'BENIGN': 0,
    'ATTACK': 1
}

# Multi-class classification
MULTICLASS_LABELS = {
    'BENIGN': 0,
    'DoS Hulk': 1,
    'DoS GoldenEye': 2,
    'DoS slowloris': 3,
    'DoS Slowhttptest': 4,
    'Heartbleed': 5,
    'DDoS': 6,
    'PortScan': 7,
    'FTP-Patator': 8,
    'SSH-Patator': 9,
    'Bot': 10,
    'Infiltration': 11,
    'Web Attack � Brute Force': 12,
    'Web Attack � XSS': 13,
    'Web Attack � Sql Injection': 14,
}

# Attack categories for grouped classification
ATTACK_CATEGORIES = {
    'BENIGN': 'Benign',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'Heartbleed': 'DoS',
    'DDoS': 'DDoS',
    'PortScan': 'Reconnaissance',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Botnet',
    'Infiltration': 'Infiltration',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
}

# ============== FEATURE CONFIGURATION ==============
# Columns to drop (non-numeric, identifiers, or duplicates)
COLUMNS_TO_DROP = [
    'Flow ID',
    'Source IP',
    'Source Port',
    'Destination IP',
    'Timestamp',
    'Fwd Header Length.1',  # Duplicate column
]

# Top features based on importance (from literature)
TOP_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Idle Mean',
    'Idle Std',
]

# ============== PREPROCESSING CONFIGURATION ==============
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Sampling configuration for handling imbalance
SAMPLE_SIZE = 500000  # Total samples to use (None for all data)
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 'auto'

# ============== MODEL HYPERPARAMETERS ==============
# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
}

# Neural Network
NN_PARAMS = {
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'early_stopping_patience': 5,
}

# LSTM
LSTM_PARAMS = {
    'units': 64,
    'dropout': 0.2,
    'recurrent_dropout': 0.2,
    'batch_size': 256,
    'epochs': 30,
    'sequence_length': 10,
}

# Autoencoder (for anomaly detection)
AE_PARAMS = {
    'encoding_dim': 16,
    'hidden_layers': [64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'contamination': 0.1,  # Expected proportion of anomalies
}

# ============== STEALTHMESH CONFIGURATION ==============
STEALTHMESH_CONFIG = {
    'alert_threshold': 0.8,  # Confidence threshold to trigger alert
    'containment_threshold': 0.95,  # Threshold for micro-containment
    'peer_broadcast_interval': 5,  # Seconds between peer updates
    'decoy_activation_attacks': ['PortScan', 'Infiltration', 'Bot'],
    'mtd_trigger_attacks': ['PortScan', 'DDoS'],
}
