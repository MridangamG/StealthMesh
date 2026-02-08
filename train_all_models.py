"""
Multi-Dataset Model Training for StealthMesh
Trains models on all 4 datasets and generates comparison results
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")


# Dataset configurations
DATASETS = {
    'CICIDS_2017': {
        'prefix': 'binary',
        'type': 'binary',
        'description': 'CICIDS 2017 - Binary (Benign vs Attack)',
        'classes': 2
    },
    'Network_MultiClass': {
        'prefix': 'network_multiclass',
        'type': 'multiclass',
        'description': 'Network Attacks - 10 Classes',
        'classes': 10
    },
    'ZeroDay': {
        'prefix': 'zeroday',
        'type': 'binary',
        'description': 'Zero-Day Attack Detection (V1→V2)',
        'classes': 2
    },
    'Ransomware': {
        'prefix': 'ransomware',
        'type': 'multiclass',
        'description': 'Ransomware/Botnet Classification',
        'classes': 3
    }
}


def load_dataset(prefix):
    """Load preprocessed dataset"""
    X_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'))
    X_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'))
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_classes):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def train_xgboost(X_train, y_train, n_classes):
    """Train XGBoost classifier"""
    if not HAS_XGBOOST:
        return None, 0
    
    objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss' if n_classes == 2 else 'mlogloss',
        objective=objective,
        n_jobs=-1
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def train_neural_network(X_train, y_train, n_classes):
    """Train Neural Network classifier"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def evaluate_model(model, X_test, y_test, n_classes):
    """Evaluate model and return metrics"""
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start) / len(X_test) * 1000  # ms per sample
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
    
    # ROC-AUC
    try:
        if n_classes == 2:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
    except Exception:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'inference_ms': inference_time
    }


def train_all_models():
    """Train all models on all datasets"""
    print("="*80)
    print("STEALTHMESH MULTI-DATASET MODEL TRAINING")
    print("="*80)
    
    all_results = []
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"DATASET: {config['description']}")
        print(f"{'='*80}")
        
        # Check if dataset exists
        try:
            X_train, X_test, y_train, y_test = load_dataset(config['prefix'])
            print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
        except FileNotFoundError:
            print(f"⚠️ Dataset not found: {config['prefix']}_*.npy - Skipping")
            continue
        
        n_classes = config['classes']
        
        # Train models
        models_to_train = [
            ('RandomForest', train_random_forest),
            ('XGBoost', train_xgboost),
            ('NeuralNetwork', train_neural_network)
        ]
        
        for model_name, train_func in models_to_train:
            print(f"\n  Training {model_name}...")
            
            try:
                model, train_time = train_func(X_train, y_train, n_classes)
                
                if model is None:
                    print(f"    ⚠️ Skipped (not available)")
                    continue
                
                # Evaluate
                metrics = evaluate_model(model, X_test, y_test, n_classes)
                metrics['train_time'] = train_time
                
                print(f"    ✓ Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"      F1-Score: {metrics['f1']*100:.2f}%")
                print(f"      ROC-AUC:  {metrics['roc_auc']*100:.2f}%")
                print(f"      Time:     {train_time:.2f}s")
                
                # Save model
                model_filename = f"{config['prefix']}_{model_name.lower()}_model.pkl"
                model_path = os.path.join(MODELS_DIR, model_filename)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"      Saved:    {model_filename}")
                
                # Store results
                all_results.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'] * 100,
                    'Precision': metrics['precision'] * 100,
                    'Recall': metrics['recall'] * 100,
                    'F1-Score': metrics['f1'] * 100,
                    'ROC-AUC': metrics['roc_auc'] * 100,
                    'Training (s)': train_time,
                    'Inference (ms)': metrics['inference_ms']
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, 'multi_dataset_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print(f"\n✓ Results saved to: {results_path}")
    
    return results_df


def generate_comparison_table():
    """Generate formatted comparison table"""
    results_path = os.path.join(RESULTS_DIR, 'multi_dataset_comparison.csv')
    
    if not os.path.exists(results_path):
        print("No results found. Run training first.")
        return
    
    df = pd.read_csv(results_path)
    
    # Pivot table for better visualization
    print("\n" + "="*80)
    print("BEST MODEL PER DATASET")
    print("="*80)
    
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        best = subset.loc[subset['Accuracy'].idxmax()]
        print(f"\n{dataset}:")
        print(f"  Best Model: {best['Model']}")
        print(f"  Accuracy:   {best['Accuracy']:.2f}%")
        print(f"  F1-Score:   {best['F1-Score']:.2f}%")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = train_all_models()
    generate_comparison_table()
