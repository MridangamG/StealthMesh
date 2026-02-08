"""
Multi-Dataset Preprocessing Pipeline for StealthMesh
Handles: CICIDS 2017, train_test_network, Zero-Day V1 & V2
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE

# Dataset paths
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "Data Sets")
CICIDS_DIR = os.path.join(os.path.dirname(__file__), "CIC-IDS-2017 Dataset")


def preprocess_train_test_network():
    """
    Preprocess train_test_network.csv dataset
    Multi-class classification: 10 attack types
    """
    print("\n" + "="*70)
    print("PREPROCESSING: train_test_network.csv")
    print("="*70)
    
    # Load data
    filepath = os.path.join(DATASETS_DIR, "train_test_network.csv")
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Show class distribution
    print("\nClass distribution:")
    print(df['type'].value_counts())
    
    # Separate features and labels
    label_col = 'type'
    
    # Drop non-numeric and identifier columns
    drop_cols = ['src_ip', 'dst_ip', 'dns_query', 'ssl_version', 'ssl_cipher', 
                 'ssl_subject', 'ssl_issuer', 'http_method', 'http_uri', 
                 'http_version', 'http_user_agent', 'http_orig_mime_types',
                 'http_resp_mime_types', 'weird_name', 'weird_addl', 'label', 'type']
    
    # Keep only columns that exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # Get labels
    y = df[label_col].copy()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"\nLabel mapping: {label_mapping}")
    
    # Get features
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Convert categorical columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Handle missing and infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"\nFinal features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)[:10]}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save
    prefix = "network_multiclass"
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'), y_test)
    
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_features.pkl'), 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n✓ Saved: {prefix}_*.npy/pkl")
    print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"  Classes: {len(label_mapping)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_mapping


def preprocess_zero_day_datasets():
    """
    Preprocess Zero-Day Attack Detection datasets
    V1 (400k) for training, V2 (97k) for testing
    Binary classification: Attack vs No Attack
    """
    print("\n" + "="*70)
    print("PREPROCESSING: Zero-Day Attack Detection Datasets")
    print("="*70)
    
    # Load V1 (training)
    v1_path = os.path.join(DATASETS_DIR, "zero_day_attack_detection_dataset_V1-400k.csv")
    print(f"\nLoading V1 (training): {v1_path}...")
    df_v1 = pd.read_csv(v1_path)
    print(f"V1: {len(df_v1)} samples")
    
    # Load V2 (testing)
    v2_path = os.path.join(DATASETS_DIR, "zero_day_attack_detection_dataset_V2_97k.csv")
    print(f"Loading V2 (testing): {v2_path}...")
    df_v2 = pd.read_csv(v2_path)
    print(f"V2: {len(df_v2)} samples")
    
    # Show distribution
    print("\nV1 Label distribution:")
    print(df_v1['Prediction'].value_counts())
    print("\nV2 Label distribution:")
    print(df_v2['Prediction'].value_counts())
    
    # Define feature columns (numeric only)
    label_col = 'Prediction'
    drop_cols = ['Time', 'SeddAddress', 'ExpAddress', 'IP Address', 'Geolocation',
                 'Logistics ID', 'Event Description', 'Session ID', 'User-Agent',
                 'Application Layer Data', 'Prediction', 'Threat Level', 'Response Time',
                 'Data Transfer Rate']
    
    # Process both datasets
    def process_df(df, name):
        # Get labels - "Attack Detected" = 1, "No Attack" = 0
        y = df[label_col].apply(lambda x: 1 if str(x).strip() == 'Attack Detected' else 0)
        
        # Get features
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Convert categorical columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing/infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"\n{name}: {X.shape[1]} features, {len(y)} samples")
        print(f"  Attack: {sum(y)}, Benign: {len(y) - sum(y)}")
        
        return X, y
    
    X_v1, y_v1 = process_df(df_v1, "V1")
    X_v2, y_v2 = process_df(df_v2, "V2")
    
    # Align features between V1 and V2
    common_cols = list(set(X_v1.columns) & set(X_v2.columns))
    print(f"\nCommon features: {len(common_cols)}")
    
    X_v1 = X_v1[common_cols]
    X_v2 = X_v2[common_cols]
    
    # Scale using V1 (training) scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_v1)
    X_test_scaled = scaler.transform(X_v2)
    
    y_train = y_v1.values
    y_test = y_v2.values
    
    # Save
    prefix = "zeroday"
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'), y_test)
    
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_features.pkl'), 'wb') as f:
        pickle.dump(common_cols, f)
    
    label_mapping = {0: 'No Attack', 1: 'Attack Detected'}
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    print(f"\n✓ Saved: {prefix}_*.npy/pkl")
    print(f"  Train (V1): {X_train_scaled.shape}")
    print(f"  Test (V2): {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_final_dataset():
    """
    Preprocess final(2).csv dataset
    Ransomware/Botnet detection
    """
    print("\n" + "="*70)
    print("PREPROCESSING: final(2).csv (Ransomware/Botnet)")
    print("="*70)
    
    filepath = os.path.join(DATASETS_DIR, "final(2).csv")
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    
    # Show distribution
    print("\nPrediction distribution:")
    print(df['Prediction'].value_counts())
    
    print("\nFamily distribution:")
    print(df['Family'].value_counts())
    
    # Use Prediction as label
    label_col = 'Prediction'
    drop_cols = ['Time', 'SeddAddress', 'ExpAddress', 'IPaddress', 'Prediction', 'Family', 'Threats']
    
    # Get labels - encode as numeric
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"\nLabel mapping: {label_mapping}")
    
    # Get features
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Convert categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"Features: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save
    prefix = "ransomware"
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'), y_test)
    
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    with open(os.path.join(PROCESSED_DIR, f'{prefix}_features.pkl'), 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n✓ Saved: {prefix}_*.npy/pkl")
    print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_mapping


def main():
    """Preprocess all datasets"""
    print("="*70)
    print("STEALTHMESH MULTI-DATASET PREPROCESSING")
    print("="*70)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. train_test_network.csv (10-class)
    preprocess_train_test_network()
    
    # 2. Zero-Day datasets (binary)
    preprocess_zero_day_datasets()
    
    # 3. final(2).csv (ransomware)
    preprocess_final_dataset()
    
    print("\n" + "="*70)
    print("ALL DATASETS PREPROCESSED!")
    print("="*70)
    
    # Summary
    print("\nPreprocessed datasets:")
    print("  1. binary_*          - CICIDS 2017 (already done)")
    print("  2. network_multiclass_* - train_test_network.csv (10-class)")
    print("  3. zeroday_*         - Zero-Day V1/V2 (binary)")
    print("  4. ransomware_*      - final(2).csv")
    
    # List files
    print(f"\nFiles in {PROCESSED_DIR}:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        size = os.path.getsize(os.path.join(PROCESSED_DIR, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
