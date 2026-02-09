"""
Generate Multi-Dataset Visualizations for Research Paper
Creates comparison plots across all 4 datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Dataset info
DATASETS = {
    'CICIDS_2017': {'prefix': 'binary', 'name': 'CICIDS 2017', 'classes': ['Benign', 'Attack']},
    'Network_10Class': {'prefix': 'network_multiclass', 'name': 'Network (10-Class)', 'classes': None},
    'Ransomware': {'prefix': 'ransomware', 'name': 'Ransomware', 'classes': None}
}


def plot_multi_dataset_comparison():
    """Create comparison bar chart across all datasets"""
    print("1. Creating multi-dataset comparison chart...")
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'multi_dataset_comparison.csv'))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy by dataset and model
    ax = axes[0, 0]
    pivot = df.pivot(index='Dataset', columns='Model', values='Accuracy')
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Datasets')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', loc='lower right')
    ax.set_ylim([95, 101])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    # F1-Score by dataset and model
    ax = axes[0, 1]
    pivot = df.pivot(index='Dataset', columns='Model', values='F1-Score')
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('F1-Score Comparison Across Datasets')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model', loc='lower right')
    ax.set_ylim([92, 101])
    
    # Training time comparison
    ax = axes[1, 0]
    pivot = df.pivot(index='Dataset', columns='Model', values='Training (s)')
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Model')
    ax.set_yscale('log')
    
    # Best model per dataset
    ax = axes[1, 1]
    best_per_dataset = df.loc[df.groupby('Dataset')['Accuracy'].idxmax()]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax.barh(best_per_dataset['Dataset'], best_per_dataset['Accuracy'], color=colors)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Best Model Performance per Dataset')
    ax.set_xlim([95, 101])
    
    # Add model names on bars
    for bar, model in zip(bars, best_per_dataset['Model']):
        ax.text(bar.get_width() - 2, bar.get_y() + bar.get_height()/2, 
                model, va='center', ha='right', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'multi_dataset_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'multi_dataset_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: multi_dataset_comparison.png/pdf")


def plot_dataset_summary():
    """Create dataset summary visualization"""
    print("2. Creating dataset summary chart...")
    
    dataset_info = {
        'Dataset': ['CICIDS 2017', 'Network 10-Class', 'Ransomware'],
        'Samples': [45365, 211043, 149043],
        'Features': [40, 27, 7],
        'Classes': [2, 10, 3],
        'Type': ['Binary', 'Multi-class', 'Multi-class']
    }
    df = pd.DataFrame(dataset_info)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample counts
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = axes[0].bar(df['Dataset'], df['Samples'], color=colors, edgecolor='white')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Dataset Size Comparison')
    axes[0].set_xticklabels(df['Dataset'], rotation=45, ha='right')
    for bar, val in zip(bars, df['Samples']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # Feature counts
    bars = axes[1].bar(df['Dataset'], df['Features'], color=colors, edgecolor='white')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title('Feature Count Comparison')
    axes[1].set_xticklabels(df['Dataset'], rotation=45, ha='right')
    for bar, val in zip(bars, df['Features']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10)
    
    # Class counts
    bars = axes[2].bar(df['Dataset'], df['Classes'], color=colors, edgecolor='white')
    axes[2].set_ylabel('Number of Classes')
    axes[2].set_title('Classification Task Complexity')
    axes[2].set_xticklabels(df['Dataset'], rotation=45, ha='right')
    for bar, val, typ in zip(bars, df['Classes'], df['Type']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val} ({typ})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_summary.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_summary.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: dataset_summary.png/pdf")


def plot_confusion_matrices_all():
    """Create confusion matrices for best model on each dataset"""
    print("3. Creating confusion matrices for all datasets...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    datasets_to_plot = [
        ('binary', 'CICIDS 2017', 'xgboost', ['Benign', 'Attack']),
        ('network_multiclass', 'Network 10-Class', 'randomforest', None),
        ('ransomware', 'Ransomware', 'randomforest', None)
    ]
    
    for idx, (prefix, title, model_name, labels) in enumerate(datasets_to_plot):
        try:
            # Load data
            X_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'))
            y_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'))
            
            # Load model
            model_path = os.path.join(MODELS_DIR, f'{prefix}_{model_name}_model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Predict
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Get labels
            if labels is None:
                label_path = os.path.join(PROCESSED_DIR, f'{prefix}_label_mapping.pkl')
                with open(label_path, 'rb') as f:
                    label_map = pickle.load(f)
                labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            
            # Plot
            if len(labels) <= 3:
                sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                           xticklabels=labels, yticklabels=labels, ax=axes[idx])
            else:
                sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=axes[idx])
                
            axes[idx].set_title(f'{title} (Best: {model_name.upper()})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[idx].set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_confusion_matrices.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'all_confusion_matrices.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: all_confusion_matrices.png/pdf")


def plot_model_ranking():
    """Create model ranking visualization"""
    print("4. Creating model ranking chart...")
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'multi_dataset_comparison.csv'))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average metrics per model
    model_avg = df.groupby('Model').agg({
        'Accuracy': 'mean',
        'F1-Score': 'mean',
        'ROC-AUC': 'mean',
        'Training (s)': 'mean'
    }).round(2)
    
    x = np.arange(len(model_avg))
    width = 0.25
    
    bars1 = ax.bar(x - width, model_avg['Accuracy'], width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x, model_avg['F1-Score'], width, label='F1-Score', color='#2ecc71')
    bars3 = ax.bar(x + width, model_avg['ROC-AUC'], width, label='ROC-AUC', color='#e74c3c')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Average Model Performance Across All Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(model_avg.index)
    ax.legend(loc='lower right')
    ax.set_ylim([95, 101])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - 1,
                   f'{height:.1f}', ha='center', va='top', fontsize=8, color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_ranking.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'model_ranking.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: model_ranking.png/pdf")


def generate_latex_tables():
    """Generate LaTeX tables for research paper"""
    print("5. Generating LaTeX tables...")
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'multi_dataset_comparison.csv'))
    
    # Table 1: Dataset Summary
    latex1 = """
\\begin{table}[htbp]
\\centering
\\caption{Dataset Summary}
\\label{tab:dataset_summary}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Features} & \\textbf{Classes} & \\textbf{Type} \\\\
\\hline
CICIDS 2017 & 45,365 & 40 & 2 & Binary \\\\
Network Attacks & 211,043 & 27 & 10 & Multi-class \\\\
Ransomware & 149,043 & 7 & 3 & Multi-class \\\\
\\hline
\\textbf{Total} & \\textbf{405,451} & - & - & - \\\\
\\hline
\\end{tabular}
\\end{table}
"""
    
    # Table 2: Model Performance
    latex2 = """
\\begin{table*}[htbp]
\\centering
\\caption{Model Performance Comparison Across All Datasets}
\\label{tab:model_performance}
\\begin{tabular}{llccccc}
\\hline
\\textbf{Dataset} & \\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{Recall (\\%)} & \\textbf{F1-Score (\\%)} & \\textbf{ROC-AUC (\\%)} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex2 += f"{row['Dataset']} & {row['Model']} & {row['Accuracy']:.2f} & {row['Precision']:.2f} & {row['Recall']:.2f} & {row['F1-Score']:.2f} & {row['ROC-AUC']:.2f} \\\\\n"
    
    latex2 += """\\hline
\\end{tabular}
\\end{table*}
"""
    
    # Save tables
    with open(os.path.join(FIGURES_DIR, 'dataset_summary_table.tex'), 'w') as f:
        f.write(latex1)
    
    with open(os.path.join(FIGURES_DIR, 'model_performance_table.tex'), 'w') as f:
        f.write(latex2)
    
    print("  Saved: dataset_summary_table.tex, model_performance_table.tex")


def main():
    print("="*60)
    print("MULTI-DATASET VISUALIZATION GENERATOR")
    print("="*60)
    
    plot_multi_dataset_comparison()
    plot_dataset_summary()
    plot_confusion_matrices_all()
    plot_model_ranking()
    generate_latex_tables()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED!")
    print(f"Output directory: {FIGURES_DIR}")
    print("="*60)
    
    # List generated files
    files = [f for f in os.listdir(FIGURES_DIR) if 'multi' in f or 'dataset' in f or 'ranking' in f or 'all_' in f]
    print(f"\nNew files generated:")
    for f in sorted(files):
        print(f"  {f}")


if __name__ == "__main__":
    main()
