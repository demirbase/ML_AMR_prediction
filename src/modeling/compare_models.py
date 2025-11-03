#!/usr/bin/env python3
"""
Model Comparison and Visualization Script

Generates comprehensive visualizations and comparisons for all trained models:
- ROC curves (individual and combined)
- Precision-Recall curves (individual and combined)
- Calibration curves (individual)
- Confusion matrices
- Metrics comparison (bar charts, heatmaps, boxplots, radar charts)
- Model ranking by F1-score

Based on the original ML_AMRprediction project's visualization approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Some visualizations will use matplotlib fallback.")

# Set style
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# ============================================================
# Configuration
# ============================================================
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("reports/comparison_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LogisticRegression": "Logistic Regression",
    "MLPClassifier": "MLP Neural Network",
    "SVM": "Support Vector Machine"
}

COLORS = {
    "RandomForest": "#2ecc71",  # Green
    "GradientBoosting": "#3498db",  # Blue
    "LogisticRegression": "#e74c3c",  # Red
    "MLPClassifier": "#9b59b6",  # Purple
    "SVM": "#f39c12"  # Orange
}

# ============================================================
# Helper Functions
# ============================================================

def load_metrics(model_name, set_name="Test"):
    """Load metrics from text file."""
    metrics_file = MODELS_DIR / model_name / f"{set_name}_metrics.txt"
    metrics = {}
    with open(metrics_file, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    metrics[key.strip()] = val.strip()
    return metrics

def load_probabilities(model_name, set_name="Test"):
    """Load probability predictions."""
    proba_file = MODELS_DIR / model_name / f"{set_name}_proba.csv"
    return pd.read_csv(proba_file)

def load_confusion_matrix(model_name, set_name="Test"):
    """Load confusion matrix."""
    cm_file = MODELS_DIR / model_name / f"{set_name}_confusion_matrix.csv"
    return pd.read_csv(cm_file, header=None).values

def load_calibration_curve(model_name, set_name="Test"):
    """Load calibration curve data."""
    calib_file = MODELS_DIR / model_name / f"{set_name}_calibration_curve.csv"
    return pd.read_csv(calib_file)

# ============================================================
# Individual Model Visualizations
# ============================================================

def plot_individual_confusion_matrix(model_name):
    """Plot confusion matrix for a single model."""
    cm = load_confusion_matrix(model_name)
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=True,
                    xticklabels=['Sensitive', 'Resistant'],
                    yticklabels=['Sensitive', 'Resistant'])
    else:
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.xticks([0, 1], ['Sensitive', 'Resistant'])
        plt.yticks([0, 1], ['Sensitive', 'Resistant'])
    plt.title(f'{MODEL_NAMES[model_name]} - Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_individual_roc_curve(model_name):
    """Plot ROC curve for a single model."""
    df_proba = load_probabilities(model_name)
    y_true = df_proba['y_true'].values
    y_score = df_proba['y_proba'].values
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=COLORS[model_name], lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{MODEL_NAMES[model_name]} - ROC Curve (Test Set)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")
    
    return fpr, tpr, roc_auc

def plot_individual_pr_curve(model_name):
    """Plot Precision-Recall curve for a single model."""
    df_proba = load_probabilities(model_name)
    y_true = df_proba['y_true'].values
    y_score = df_proba['y_proba'].values
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color=COLORS[model_name], lw=3,
             label=f'PR Curve (AP = {ap_score:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'{MODEL_NAMES[model_name]} - Precision-Recall Curve (Test Set)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"{model_name}_pr_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")
    
    return precision, recall, ap_score

def plot_individual_calibration(model_name):
    """Plot calibration curve for a single model."""
    df_calib = load_calibration_curve(model_name)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_calib['prob_pred'], df_calib['prob_true'],
             marker='o', linewidth=3, color=COLORS[model_name], 
             label=f'{MODEL_NAMES[model_name]}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.title(f'{MODEL_NAMES[model_name]} - Calibration Curve (Test Set)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"{model_name}_calibration_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

# ============================================================
# Combined Visualizations
# ============================================================

def plot_combined_roc_curves(roc_data):
    """Plot ROC curves for all models on one chart."""
    plt.figure(figsize=(10, 8))
    
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=3, color=COLORS[model_name],
                label=f'{MODEL_NAMES[model_name]} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves - All Models Comparison (Test Set)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "Test_roc_curve_all_models.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_combined_pr_curves(pr_data):
    """Plot Precision-Recall curves for all models on one chart."""
    plt.figure(figsize=(10, 8))
    
    for model_name, (precision, recall, ap_score) in pr_data.items():
        plt.plot(recall, precision, lw=3, color=COLORS[model_name],
                label=f'{MODEL_NAMES[model_name]} (AP = {ap_score:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves - All Models Comparison (Test Set)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "Test_pr_curve_all_models.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

# ============================================================
# Metrics Comparison Visualizations
# ============================================================

def create_metrics_dataframe():
    """Load all metrics into a single DataFrame."""
    all_metrics = []
    for model_name in MODEL_NAMES.keys():
        metrics = load_metrics(model_name, "Test")
        metrics['Model'] = MODEL_NAMES[model_name]
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    # Reorder columns
    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'mcc', 'brier_score']
    df = df[['Model'] + metric_cols]
    
    # Save to CSV
    output_file = OUTPUT_DIR / "all_models_metrics.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    return df

def plot_metrics_bar_chart(df_metrics):
    """Create bar chart comparing key metrics across models."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    df_plot = df_metrics[['Model'] + metrics_to_plot].melt(id_vars='Model', 
                                                             var_name='Metric', 
                                                             value_name='Score')
    
    plt.figure(figsize=(14, 8))
    if HAS_SEABORN:
        sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', palette=COLORS.values())
    else:
        # Fallback to matplotlib grouped bar chart
        x = np.arange(len(metrics_to_plot))
        width = 0.15
        for i, model in enumerate(df_metrics['Model']):
            model_key = [k for k, v in MODEL_NAMES.items() if v == model][0]
            values = df_metrics.loc[df_metrics['Model'] == model, metrics_to_plot].values[0]
            plt.bar(x + i * width, values, width, label=model, color=COLORS[model_key])
        plt.xticks(x + width * 2, metrics_to_plot)
    
    plt.title('Performance Metrics Comparison - All Models (Test Set)', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "Test_bar_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_metrics_heatmap(df_metrics):
    """Create heatmap of all metrics."""
    df_plot = df_metrics.set_index('Model')
    
    plt.figure(figsize=(12, 8))
    if HAS_SEABORN:
        sns.heatmap(df_plot, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
    else:
        im = plt.imshow(df_plot.values, cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, label='Score')
        plt.xticks(range(len(df_plot.columns)), df_plot.columns, rotation=45)
        plt.yticks(range(len(df_plot.index)), df_plot.index)
        for i in range(len(df_plot.index)):
            for j in range(len(df_plot.columns)):
                plt.text(j, i, f'{df_plot.values[i, j]:.4f}', 
                        ha='center', va='center', color='black', fontsize=9)
    plt.title('Metrics Heatmap - All Models (Test Set)', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "Test_metrics_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_metrics_boxplot(df_metrics):
    """Create boxplot showing distribution of metrics."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'mcc']
    
    df_plot = df_metrics[metrics_to_plot]
    
    plt.figure(figsize=(14, 8))
    bp = plt.boxplot([df_plot[col].values for col in metrics_to_plot],
                      labels=metrics_to_plot,
                      patch_artist=True,
                      showmeans=True,
                      meanline=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    plt.title('Metrics Distribution Across Models (Test Set)', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "Test_metrics_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_radar_chart(df_metrics):
    """Create radar chart for model comparison."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'mcc']
    
    # Number of variables
    num_vars = len(metrics_to_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for idx, row in df_metrics.iterrows():
        model_name = [k for k, v in MODEL_NAMES.items() if v == row['Model']][0]
        values = row[metrics_to_plot].values.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], 
                color=COLORS[model_name])
        ax.fill(angles, values, alpha=0.15, color=COLORS[model_name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.title('Model Performance Radar Chart (Test Set)', fontsize=16, fontweight='bold', 
              pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "radar_chart_Test.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def create_model_ranking(df_metrics):
    """Create model ranking based on F1-score."""
    df_ranked = df_metrics.sort_values('f1_score', ascending=False)
    
    ranking_text = """=== Model Ranking (by F1-Score) ===

F1-Score balances precision and recall, making it the key metric for AMR prediction.
Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)

"""
    
    for idx, (i, row) in enumerate(df_ranked.iterrows(), 1):
        ranking_text += f"{idx}. {row['Model']:<25} → F1-Score = {row['f1_score']:.4f} "
        ranking_text += f"(Accuracy: {row['accuracy']:.4f})\n"
    
    output_file = OUTPUT_DIR / "model_ranking.txt"
    with open(output_file, 'w') as f:
        f.write(ranking_text)
    print(f"✓ Saved: {output_file}")
    
    return df_ranked

# ============================================================
# Main Execution
# ============================================================

def main():
    print("="*60)
    print("Model Comparison & Visualization")
    print("="*60)
    print()
    
    # Individual model visualizations
    print("Generating individual model visualizations...")
    roc_data = {}
    pr_data = {}
    
    for model_name in MODEL_NAMES.keys():
        print(f"\n  Processing {MODEL_NAMES[model_name]}...")
        plot_individual_confusion_matrix(model_name)
        fpr, tpr, roc_auc = plot_individual_roc_curve(model_name)
        precision, recall, ap_score = plot_individual_pr_curve(model_name)
        plot_individual_calibration(model_name)
        
        roc_data[model_name] = (fpr, tpr, roc_auc)
        pr_data[model_name] = (precision, recall, ap_score)
    
    # Combined visualizations
    print("\n\nGenerating combined visualizations...")
    plot_combined_roc_curves(roc_data)
    plot_combined_pr_curves(pr_data)
    
    # Metrics comparison
    print("\nGenerating metrics comparison...")
    df_metrics = create_metrics_dataframe()
    plot_metrics_bar_chart(df_metrics)
    plot_metrics_heatmap(df_metrics)
    plot_metrics_boxplot(df_metrics)
    plot_radar_chart(df_metrics)
    create_model_ranking(df_metrics)
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
