# StealthMesh: Adaptive Stealth Communication and Decentralized Defense

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

**StealthMesh** is a stealth-enabled, decentralized cyber defense framework tailored for Micro, Small, and Medium Enterprises (MSMEs). The system integrates advanced stealth communication techniques with a decentralized mesh defense protocol to provide affordable, lightweight, and adaptive protection against sophisticated cyber threats.

### Key Features

- 🔐 **Polymorphic Encryption** - Rotating cipher algorithms to prevent pattern analysis
- 🛡️ **Decoy Routing** - Dynamic path selection with fake traffic injection
- 🕸️ **Mesh Network Defense** - Peer alerts, consensus voting, coordinated response
- 🤖 **ML-Based Threat Detection** - Real-time attack classification (99.63% accuracy)
- 📦 **Micro-Containment** - Autonomous breach isolation and quarantine
- 🎯 **Adaptive MTD** - Moving Target Defense with dynamic port/service mutation

## 📊 Model Performance (Multi-Dataset Evaluation)

### Dataset Summary

| Dataset          | Samples     | Features | Classes | Type        |
| ---------------- | ----------- | -------- | ------- | ----------- |
| CICIDS 2017      | 45,365      | 40       | 2       | Binary      |
| Network 10-Class | 211,043     | 27       | 10      | Multi-class |
| Ransomware       | 149,043     | 7        | 3       | Multi-class |
| **Total**        | **405,451** | -        | -       | -           |

### Best Model Performance per Dataset

| Dataset          | Best Model   | Accuracy   | F1-Score | ROC-AUC |
| ---------------- | ------------ | ---------- | -------- | ------- |
| CICIDS 2017      | XGBoost      | **99.58%** | 99.27%   | 99.86%  |
| Network 10-Class | RandomForest | **98.94%** | 97.48%   | 99.92%  |
| Ransomware       | RandomForest | **97.73%** | 93.83%   | 99.44%  |

### Detailed Results (All Models)

| Dataset         | Model            | Accuracy   | Precision | Recall | F1-Score | ROC-AUC |
| --------------- | ---------------- | ---------- | --------- | ------ | -------- | ------- |
| CICIDS_2017     | RandomForest     | 99.47%     | 99.19%    | 99.47% | 99.18%   | 99.86%  |
| CICIDS_2017     | **XGBoost**      | **99.58%** | 99.39%    | 99.58% | 99.27%   | 99.86%  |
| CICIDS_2017     | NeuralNetwork    | 97.06%     | 96.63%    | 97.06% | 96.50%   | 99.48%  |
| Network_10Class | **RandomForest** | **98.94%** | 97.47%    | 98.94% | 97.48%   | 99.92%  |
| Network_10Class | XGBoost          | 98.94%     | 97.45%    | 98.94% | 97.47%   | 99.92%  |
| Network_10Class | NeuralNetwork    | 98.53%     | 96.76%    | 98.53% | 97.05%   | 99.89%  |
| Ransomware      | **RandomForest** | **97.73%** | 96.42%    | 97.73% | 93.83%   | 99.44%  |
| Ransomware      | XGBoost          | 97.46%     | 96.09%    | 97.46% | 93.33%   | 99.39%  |
| Ransomware      | NeuralNetwork    | 96.99%     | 95.75%    | 96.99% | 92.15%   | 99.32%  |

## 🏗️ Project Structure

```
StealthMesh/
├── CIC-IDS-2017 Dataset/          # CICIDS 2017 dataset (8 CSV files)
├── Data Sets/                     # Additional datasets
│   ├── train_test_network.csv     # 10-class network attacks (211k samples)
│   └── final(2).csv               # Ransomware/Botnet (149k samples)
├── src/
│   ├── preprocessing/             # Data preprocessing pipeline
│   │   ├── data_loader.py         # Load and merge dataset files
│   │   ├── data_cleaner.py        # Clean, encode, handle missing values
│   │   ├── feature_engineer.py    # Feature selection and scaling
│   │   └── data_pipeline.py       # Complete preprocessing pipeline
│   ├── models/                    # ML model implementations
│   │   ├── base_model.py          # Abstract base class
│   │   ├── random_forest.py       # Random Forest classifier
│   │   ├── xgboost_model.py       # XGBoost classifier
│   │   └── neural_network.py      # MLP Neural Network
│   └── stealthmesh/               # Core defense modules
│       ├── stealth_comm.py        # Stealth Communication Engine
│       ├── decoy_routing.py       # Decoy Routing Module
│       ├── mesh_coordinator.py    # Mesh Network Coordinator
│       ├── threat_detector.py     # ML-Based Threat Detection
│       ├── micro_containment.py   # Micro-Containment Engine
│       ├── adaptive_mtd.py        # Adaptive Moving Target Defense
│       └── stealthmesh_node.py    # Integrated Defense Node
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis
├── processed_data/                # Preprocessed data files
│   ├── binary_*.npy               # CICIDS 2017 binary classification
│   ├── network_multiclass_*.npy   # Network 10-class classification
│   └── ransomware_*.npy           # Ransomware detection
├── models/                        # Trained model files (12 models)
│   ├── binary_*_model.pkl         # CICIDS 2017 models
│   ├── network_multiclass_*_model.pkl  # Network attack models
│   └── ransomware_*_model.pkl     # Ransomware detection models
├── results/
│   ├── model_comparison.csv       # CICIDS 2017 results
│   ├── multi_dataset_comparison.csv  # All datasets comparison
│   └── figures/                   # Visualizations for paper (26 files)
│       ├── multi_dataset_comparison.png/pdf
│       ├── all_confusion_matrices.png/pdf
│       ├── dataset_summary.png/pdf
│       ├── model_ranking.png/pdf
│       └── ... (more figures)
├── config.py                      # Project configuration
├── preprocess_data.py             # CICIDS 2017 preprocessing
├── preprocess_all_datasets.py     # Multi-dataset preprocessing
├── train_models.py                # CICIDS 2017 training
├── train_all_models.py            # Multi-dataset training
├── demo_stealthmesh.py            # Full demonstration
├── generate_visualizations.py     # CICIDS 2017 figures
├── generate_multi_dataset_viz.py  # Multi-dataset figures
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
python preprocess_data.py --classification binary --features 40
```

### 3. Train Models

```bash
python train_models.py
```

### 4. Run Demo

```bash
python demo_stealthmesh.py
```

### 5. Generate Visualizations

```bash
python generate_visualizations.py
```

## 📚 Dataset Information

This project uses **3 datasets** for comprehensive cyber threat detection evaluation:

### 1. CICIDS 2017 (Binary Classification)

**Canadian Institute for Cybersecurity Intrusion Detection Dataset**

| Property       | Value                           |
| -------------- | ------------------------------- |
| Total Records  | ~2.83 Million (sampled: 45,365) |
| Features       | 78 (selected: 40)               |
| Attack Types   | 14                              |
| Days Covered   | 5 (Mon-Fri)                     |
| Binary Classes | Benign (80.3%), Attack (19.7%)  |

**Attack Types:** DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, Web Attack (Brute Force, XSS, SQL Injection), Infiltration, Heartbleed

### 2. Network 10-Class (Multi-class Classification)

**Network Intrusion Detection Dataset**

| Property      | Value   |
| ------------- | ------- |
| Total Records | 211,043 |
| Features      | 27      |
| Classes       | 10      |

**Attack Classes:** backdoor, ddos, dos, injection, mitm, normal, password, ransomware, scanning, xss

### 3. Ransomware/Botnet Detection (Multi-class Classification)

**Ransomware and Botnet Family Classification**

| Property      | Value      |
| ------------- | ---------- |
| Total Records | 149,043    |
| Features      | 7          |
| Classes       | 3 families |

**Attack Families:** Ransomware, Botnet, Normal

## 🔬 StealthMesh Modules

### 1. Stealth Communication Engine

- **Polymorphic Encryption**: AES-256-GCM/CBC with automatic cipher rotation
- **Packet Camouflage**: HTTP/DNS/TLS traffic mimicry
- **Covert Channels**: Hidden data transmission techniques

### 2. Decoy Routing Module

- **Dynamic Path Selection**: Probabilistic multi-hop routing
- **Fake Traffic Injection**: Chaff, padding, and decoy alerts
- **Route Obfuscation**: Prevents traffic analysis attacks

### 3. Mesh Network Coordinator

- **Peer Discovery**: Automatic node registration
- **Gossip Protocol**: Efficient alert propagation
- **Consensus Voting**: Byzantine fault-tolerant decisions
- **Trust Scoring**: Dynamic peer reliability assessment

### 4. Threat Detection Engine

- **ML Classification**: XGBoost, Random Forest, Neural Network
- **Real-time Analysis**: <10ms inference time per flow
- **Confidence Scoring**: Probabilistic threat assessment
- **Attack Type Classification**: Multi-class detection capability

### 5. Micro-Containment Engine

- **Auto-Escalation**: Progressive response based on offense count
- **Rule-based Isolation**: Port blocking, IP blocking, quarantine
- **Whitelist/Blacklist**: Trusted and known-bad IP management
- **Dry-run Mode**: Safe testing without actual network changes

### 6. Adaptive MTD (Moving Target Defense)

- **Port Mutation**: Dynamic service port shuffling
- **Honeypot Deployment**: Decoy services to deceive attackers
- **Emergency Response**: Rapid surface mutation under attack
- **Threat-level Adaptation**: Response intensity scaling

## 📈 Research Paper Structure

```
1. ABSTRACT
   - Problem statement, solution overview, key results

2. INTRODUCTION
   - MSME cybersecurity challenges
   - Motivation and research gap
   - Contributions

3. RELATED WORK
   - Moving Target Defense (MTD)
   - Stealth communication techniques
   - Decentralized security frameworks
   - ML-based intrusion detection

4. DATASET DESCRIPTION
   - CICIDS 2017 overview
   - Feature analysis
   - Class distribution
   - Preprocessing methodology

5. PROPOSED METHODOLOGY
   - StealthMesh architecture
   - Module design and algorithms
   - Integration approach

6. IMPLEMENTATION
   - Technology stack
   - ML model training
   - Defense module implementation

7. EXPERIMENTAL RESULTS
   - Model performance comparison
   - Confusion matrices and ROC curves
   - Feature importance analysis
   - System benchmarks

8. DISCUSSION
   - Comparison with existing solutions
   - Limitations and future work
   - Real-world applicability

9. CONCLUSION

10. REFERENCES
```

## 📊 Generated Figures for Paper

All figures are saved in `results/figures/` in both PNG and PDF formats:

| Figure                         | Description                       |
| ------------------------------ | --------------------------------- |
| `model_comparison.png`         | Performance metrics bar chart     |
| `confusion_matrices.png`       | Confusion matrices for all models |
| `roc_curves.png`               | ROC curves comparison             |
| `precision_recall_curves.png`  | PR curves comparison              |
| `dataset_distribution.png`     | CICIDS 2017 class distribution    |
| `feature_importance.png`       | Top 15 important features         |
| `stealthmesh_architecture.png` | System architecture diagram       |
| `attack_response_flow.png`     | Detection and response flow       |
| `performance_table.tex`        | LaTeX table for paper             |

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Paths
DATA_DIR = "CIC-IDS-2017 Dataset"
PROCESSED_DIR = "processed_data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model parameters
RF_PARAMS = {"n_estimators": 100, "max_depth": 20}
XGB_PARAMS = {"n_estimators": 100, "max_depth": 10}
NN_PARAMS = {"hidden_layer_sizes": (128, 64, 32)}
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{stealthmesh2026,
  title={StealthMesh: Adaptive Stealth Communication and Decentralized Defense Techniques for MSME Cybersecurity},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**StealthMesh** - Empowering MSMEs with affordable, adaptive cyber defense.
