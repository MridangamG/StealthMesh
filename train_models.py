"""
Main Model Training Script for StealthMesh
Train and evaluate multiple models on CICIDS 2017 dataset
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
from src.preprocessing.data_pipeline import DataPipeline
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel


def load_processed_data(classification: str = 'binary'):
    """Load preprocessed data"""
    prefix = classification
    
    X_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'))
    X_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'))
    
    print(f"Loaded data: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test, models_to_train=None):
    """
    Train and evaluate multiple models
    
    Args:
        X_train, X_test, y_train, y_test: Data splits
        models_to_train: List of model names to train
        
    Returns:
        Dictionary of trained models and their metrics
    """
    if models_to_train is None:
        models_to_train = ['rf', 'xgb', 'nn']
    
    results = {}
    
    # 1. Random Forest
    if 'rf' in models_to_train:
        print("\n" + "="*70)
        print("Training Random Forest...")
        print("="*70)
        
        rf_model = RandomForestModel()
        rf_model.build_model(n_estimators=100, max_depth=20)
        rf_model.train(X_train, y_train)
        rf_metrics = rf_model.print_evaluation(X_test, y_test)
        rf_model.save_model()
        
        results['RandomForest'] = {
            'model': rf_model,
            'metrics': rf_metrics
        }
    
    # 2. XGBoost
    if 'xgb' in models_to_train:
        print("\n" + "="*70)
        print("Training XGBoost...")
        print("="*70)
        
        try:
            xgb_model = XGBoostModel()
            xgb_model.build_model(n_estimators=100, max_depth=10)
            xgb_model.train(X_train, y_train)
            xgb_metrics = xgb_model.print_evaluation(X_test, y_test)
            xgb_model.save_model()
            
            results['XGBoost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics
            }
        except ImportError:
            print("XGBoost not installed, skipping...")
    
    # 3. Neural Network
    if 'nn' in models_to_train:
        print("\n" + "="*70)
        print("Training Neural Network...")
        print("="*70)
        
        nn_model = NeuralNetworkModel()
        nn_model.build_model(hidden_layer_sizes=(128, 64, 32))
        nn_model.train(X_train, y_train)
        nn_metrics = nn_model.print_evaluation(X_test, y_test)
        nn_model.save_model()
        
        results['NeuralNetwork'] = {
            'model': nn_model,
            'metrics': nn_metrics
        }
    
    return results


def create_comparison_report(results: dict, save_dir: str = RESULTS_DIR):
    """
    Create comparison report and visualizations
    
    Args:
        results: Dictionary of model results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, data in results.items():
        metrics = data['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'] * 100,
            'Precision': metrics['precision'] * 100,
            'Recall': metrics['recall'] * 100,
            'F1-Score': metrics['f1_score'] * 100,
            'ROC-AUC': metrics.get('roc_auc', 0) * 100,
            'Inference (ms)': metrics['inference_time_ms'],
            'Training (s)': data['model'].training_time
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'model_comparison.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nComparison saved to {csv_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Accuracy comparison
    ax1 = axes[0]
    colors = sns.color_palette('husl', len(df_comparison))
    bars = ax1.bar(df_comparison['Model'], df_comparison['Accuracy'], color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim([80, 100])
    for bar, acc in zip(bars, df_comparison['Accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. Multiple metrics comparison
    ax2 = axes[1]
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df_comparison['Model']))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        ax2.bar(x + i*width, df_comparison[metric], width, label=metric)
    
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Precision, Recall, F1 Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(df_comparison['Model'])
    ax2.legend()
    ax2.set_ylim([80, 100])
    
    # 3. Inference time comparison
    ax3 = axes[2]
    bars = ax3.bar(df_comparison['Model'], df_comparison['Inference (ms)'], color=colors)
    ax3.set_ylabel('Time (ms/sample)')
    ax3.set_title('Inference Time Comparison')
    for bar, time_val in zip(bars, df_comparison['Inference (ms)']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {fig_path}")
    plt.show()
    
    # Save confusion matrices
    for model_name, data in results.items():
        cm = data['metrics']['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(save_dir, f'{model_name.lower()}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_comparison


def main():
    parser = argparse.ArgumentParser(description='StealthMesh Model Training')
    
    parser.add_argument(
        '--classification', 
        type=str, 
        default='binary',
        choices=['binary', 'multiclass', 'category'],
        help='Type of classification (default: binary)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['rf', 'xgb', 'nn'],
        choices=['rf', 'xgb', 'nn'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run preprocessing before training'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=100000,
        help='Number of samples for preprocessing'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("   STEALTHMESH MODEL TRAINING")
    print("   Adaptive Stealth Communication and Decentralized Defense")
    print("="*70)
    
    # Preprocess if needed
    if args.preprocess:
        print("\nRunning preprocessing...")
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test = pipeline.run_pipeline(
            sample_size=args.samples,
            classification_type=args.classification,
            n_features=40,
            scale=True,
            balance=True
        )
        pipeline.save_processed_data(X_train, X_test, y_train, y_test, 
                                     prefix=args.classification)
    else:
        # Load existing preprocessed data
        try:
            X_train, X_test, y_train, y_test = load_processed_data(args.classification)
        except FileNotFoundError:
            print("Preprocessed data not found. Running preprocessing...")
            pipeline = DataPipeline()
            X_train, X_test, y_train, y_test = pipeline.run_pipeline(
                sample_size=args.samples,
                classification_type=args.classification,
                n_features=40,
                scale=True,
                balance=True
            )
            pipeline.save_processed_data(X_train, X_test, y_train, y_test,
                                         prefix=args.classification)
    
    # Train models
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        models_to_train=args.models
    )
    
    # Create comparison report
    comparison = create_comparison_report(results)
    
    print("\n" + "="*70)
    print("   TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTrained models saved to: {MODELS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
