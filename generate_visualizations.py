"""
Generate Visualizations for StealthMesh Research Paper
Creates all necessary plots and figures for publication
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import pickle

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data_and_models():
    """Load preprocessed data and trained models"""
    print("Loading data and models...")
    
    # Load test data
    X_test = np.load(os.path.join(PROCESSED_DIR, 'binary_X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'binary_y_test.npy'))
    
    # Load models
    models = {}
    model_files = {
        'Random Forest': 'randomforest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'Neural Network': 'neuralnetwork_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    models[name] = data.get('model')
                else:
                    models[name] = data
            print(f"  Loaded {name}")
    
    # Load scaler
    scaler_path = os.path.join(PROCESSED_DIR, 'binary_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return X_test, y_test, models, scaler


def plot_model_comparison():
    """Create model comparison bar chart"""
    print("\n1. Creating model comparison chart...")
    
    # Read results
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df['Model']))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i*width, df[metric], width, label=metric)
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Score (%)')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(df['Model'])
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([95, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Training time and inference time
    ax2 = axes[1]
    x = np.arange(len(df['Model']))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['Training (s)'], width, label='Training Time (s)', color='steelblue')
    ax2.set_ylabel('Training Time (seconds)', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    
    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + width/2, df['Inference (ms)'] * 1000, width, label='Inference Time (μs)', color='coral')
    ax3.set_ylabel('Inference Time (μs)', color='coral')
    ax3.tick_params(axis='y', labelcolor='coral')
    
    ax2.set_xlabel('Model')
    ax2.set_title('Training and Inference Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'])
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: model_comparison.png/pdf")


def plot_confusion_matrices(X_test, y_test, models):
    """Create confusion matrix for each model"""
    print("\n2. Creating confusion matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    labels = ['Benign', 'Attack']
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[idx],
                    cbar_kws={'label': '%'})
        axes[idx].set_title(f'{name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        
        # Add raw counts as secondary annotation
        for i in range(2):
            for j in range(2):
                axes[idx].text(j+0.5, i+0.7, f'(n={cm[i,j]})', 
                              ha='center', va='center', fontsize=8, color='gray')
    
    plt.suptitle('Confusion Matrices (Normalized %)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrices.png/pdf")


def plot_roc_curves(X_test, y_test, models):
    """Create ROC curves for all models"""
    print("\n3. Creating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (name, model) in enumerate(models.items()):
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: roc_curves.png/pdf")


def plot_precision_recall_curves(X_test, y_test, models):
    """Create Precision-Recall curves"""
    print("\n4. Creating Precision-Recall curves...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color=colors[idx], lw=2,
                label=f'{name} (AP = {ap:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'precision_recall_curves.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'precision_recall_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: precision_recall_curves.png/pdf")


def plot_dataset_distribution():
    """Plot CICIDS 2017 dataset class distribution"""
    print("\n5. Creating dataset distribution chart...")
    
    # Attack distribution from dataset analysis
    attack_data = {
        'BENIGN': 2273097,
        'DoS Hulk': 231073,
        'PortScan': 158930,
        'DDoS': 128027,
        'DoS GoldenEye': 10293,
        'FTP-Patator': 7938,
        'SSH-Patator': 5897,
        'DoS slowloris': 5796,
        'DoS Slowhttptest': 5499,
        'Bot': 1966,
        'Web Attack - Brute Force': 1507,
        'Web Attack - XSS': 652,
        'Infiltration': 36,
        'Web Attack - SQL Injection': 21,
        'Heartbleed': 11
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full distribution (log scale)
    labels = list(attack_data.keys())
    values = list(attack_data.values())
    colors = ['#2ecc71'] + ['#e74c3c'] * (len(labels) - 1)
    
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Number of Samples (log scale)')
    axes[0].set_title('CICIDS 2017 Dataset - Class Distribution')
    axes[0].invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, values):
        axes[0].text(val * 1.1, bar.get_y() + bar.get_height()/2, 
                    f'{val:,}', va='center', fontsize=8)
    
    # Binary distribution (pie chart)
    binary_data = {
        'Benign': 2273097,
        'Attack': sum(values) - 2273097
    }
    
    axes[1].pie(binary_data.values(), labels=binary_data.keys(), 
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                explode=(0, 0.05), shadow=True, startangle=90)
    axes[1].set_title('Binary Classification Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_distribution.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_distribution.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: dataset_distribution.png/pdf")


def plot_stealthmesh_architecture():
    """Create StealthMesh architecture diagram"""
    print("\n6. Creating architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'node': '#3498db',
        'module': '#2ecc71',
        'data': '#f39c12',
        'attack': '#e74c3c',
        'arrow': '#7f8c8d'
    }
    
    # Title
    ax.text(7, 9.5, 'StealthMesh Architecture', fontsize=16, fontweight='bold',
            ha='center', va='center')
    
    # Main node box
    main_box = plt.Rectangle((1, 1), 12, 7.5, fill=False, edgecolor='#2c3e50', 
                              linewidth=2, linestyle='-')
    ax.add_patch(main_box)
    ax.text(7, 8.2, 'StealthMesh Defense Node', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2c3e50')
    
    # Module boxes
    modules = [
        ('Stealth\nCommunication', 1.5, 6, '#3498db'),
        ('Decoy\nRouting', 4, 6, '#9b59b6'),
        ('Mesh\nCoordinator', 6.5, 6, '#1abc9c'),
        ('Threat\nDetector', 9, 6, '#e74c3c'),
        ('Micro\nContainment', 11.5, 6, '#e67e22'),
        ('Adaptive\nMTD', 6.5, 3.5, '#2ecc71'),
    ]
    
    for name, x, y, color in modules:
        box = plt.Rectangle((x-1, y-0.8), 2, 1.6, fill=True, 
                            facecolor=color, edgecolor='white', 
                            linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
    
    # ML Models box
    ml_box = plt.Rectangle((8, 3.5-0.8), 4.5, 1.6, fill=True,
                           facecolor='#34495e', edgecolor='white',
                           linewidth=2, alpha=0.9)
    ax.add_patch(ml_box)
    ax.text(10.25, 3.5, 'ML Models\n(RF, XGBoost, NN)', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Input/Output
    ax.annotate('', xy=(1, 4.5), xytext=(-0.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.text(-0.3, 5, 'Network\nTraffic', fontsize=9, ha='center', va='center')
    
    ax.annotate('', xy=(14.5, 4.5), xytext=(13, 4.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.text(14.3, 5, 'Defense\nActions', fontsize=9, ha='center', va='center')
    
    # Connections
    connections = [
        ((1.5, 5.2), (1.5, 4.3)),  # Stealth to center
        ((4, 5.2), (4, 4.3)),      # Decoy to center
        ((6.5, 5.2), (6.5, 4.3)),  # Mesh to center
        ((9, 5.2), (9, 4.3)),      # Threat to center
        ((11.5, 5.2), (11.5, 4.3)), # Containment to center
        ((9, 4.3), (8, 3.5)),      # Threat to ML
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#bdc3c7', lw=1.5))
    
    # Legend
    legend_items = [
        ('Stealth Communication', '#3498db'),
        ('Decoy Routing', '#9b59b6'),
        ('Mesh Coordination', '#1abc9c'),
        ('Threat Detection', '#e74c3c'),
        ('Micro-Containment', '#e67e22'),
        ('Adaptive MTD', '#2ecc71'),
    ]
    
    for i, (name, color) in enumerate(legend_items):
        x = 1.5 + (i % 3) * 4.2
        y = 0.5 - (i // 3) * 0.4
        ax.add_patch(plt.Rectangle((x-0.15, y-0.1), 0.3, 0.2, 
                                   facecolor=color, edgecolor='none'))
        ax.text(x + 0.3, y, name, fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'stealthmesh_architecture.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGURES_DIR, 'stealthmesh_architecture.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: stealthmesh_architecture.png/pdf")


def plot_feature_importance(X_test, y_test, models):
    """Plot feature importance for tree-based models"""
    print("\n7. Creating feature importance chart...")
    
    # Load feature names
    features_path = os.path.join(PROCESSED_DIR, 'binary_features.pkl')
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, ax) in enumerate([('Random Forest', axes[0]), ('XGBoost', axes[1])]):
        if name in models:
            model = models[name]
            importances = model.feature_importances_
            
            # Get top 15 features
            indices = np.argsort(importances)[-15:]
            top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                           for i in indices]
            top_importances = importances[indices]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            ax.barh(top_features, top_importances, color=colors)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{name} - Top 15 Features')
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: feature_importance.png/pdf")


def plot_attack_response_flow():
    """Create attack detection and response flow diagram"""
    print("\n8. Creating attack response flow diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'StealthMesh Attack Detection & Response Flow', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Flow boxes
    boxes = [
        ('Network Traffic\nIngress', 1, 6, '#3498db'),
        ('Threat Detection\n(ML Classification)', 4, 6, '#e74c3c'),
        ('Risk Assessment\n& Scoring', 7, 6, '#f39c12'),
        ('Response\nSelection', 10, 6, '#9b59b6'),
        ('Peer Alert\nPropagation', 2.5, 3.5, '#1abc9c'),
        ('Micro-Containment\n(Auto-Quarantine)', 6, 3.5, '#e67e22'),
        ('Adaptive MTD\n(Surface Mutation)', 9.5, 3.5, '#2ecc71'),
        ('Threat\nNeutralized', 6, 1, '#27ae60'),
    ]
    
    for name, x, y, color in boxes:
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x-1, y-0.5), 2, 1,
                            facecolor=color, edgecolor='white',
                            linewidth=2, alpha=0.85,
                            boxstyle='round,pad=0.05')
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Arrows
    arrows = [
        ((2, 6), (3, 6)),       # Traffic -> Detection
        ((5, 6), (6, 6)),       # Detection -> Assessment
        ((8, 6), (9, 6)),       # Assessment -> Response
        ((4, 5.5), (2.5, 4)),   # Detection -> Peer Alert
        ((7, 5.5), (6, 4)),     # Assessment -> Containment
        ((10, 5.5), (9.5, 4)),  # Response -> MTD
        ((2.5, 3), (5, 1.5)),   # Peer -> Neutralized
        ((6, 3), (6, 1.5)),     # Containment -> Neutralized
        ((9.5, 3), (7, 1.5)),   # MTD -> Neutralized
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Labels on arrows
    ax.text(2.5, 6.3, 'Analyze', fontsize=8, ha='center')
    ax.text(5.5, 6.3, 'Score', fontsize=8, ha='center')
    ax.text(8.5, 6.3, 'Decide', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'attack_response_flow.png'), 
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIGURES_DIR, 'attack_response_flow.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: attack_response_flow.png/pdf")


def generate_performance_table():
    """Generate LaTeX table for research paper"""
    print("\n9. Generating LaTeX performance table...")
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Model Performance Comparison on CICIDS 2017 Dataset}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{ROC-AUC} & \\textbf{Time (s)} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex_table += f"{row['Model']} & {row['Accuracy']:.2f}\\% & {row['Precision']:.2f}\\% & "
        latex_table += f"{row['Recall']:.2f}\\% & {row['F1-Score']:.2f}\\% & {row['ROC-AUC']:.2f}\\% & {row['Training (s)']:.2f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
    
    # Save to file
    with open(os.path.join(FIGURES_DIR, 'performance_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print("  Saved: performance_table.tex")
    print("\n  LaTeX Table Preview:")
    print(latex_table)


def main():
    """Generate all visualizations"""
    print("="*60)
    print("STEALTHMESH VISUALIZATION GENERATOR")
    print("="*60)
    
    # Load data and models
    X_test, y_test, models, scaler = load_data_and_models()
    
    # Generate all plots
    plot_model_comparison()
    plot_confusion_matrices(X_test, y_test, models)
    plot_roc_curves(X_test, y_test, models)
    plot_precision_recall_curves(X_test, y_test, models)
    plot_dataset_distribution()
    plot_stealthmesh_architecture()
    plot_feature_importance(X_test, y_test, models)
    plot_attack_response_flow()
    generate_performance_table()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED!")
    print(f"Output directory: {FIGURES_DIR}")
    print("="*60)
    
    # List generated files
    files = os.listdir(FIGURES_DIR)
    print(f"\nGenerated {len(files)} files:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
