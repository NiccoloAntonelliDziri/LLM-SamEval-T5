#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def set_style():
    """Set a publication-quality style for matplotlib."""
    # Try to use a seaborn style if available, otherwise fallback to a clean style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def plot_accuracy(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy for all models (Zero-shot and Five-shot)."""
    cols = ['model', 'zero_accuracy', 'five_accuracy']
    plot_df = df[cols].copy()
    plot_df = plot_df.sort_values('zero_accuracy', ascending=True)
    
    models = plot_df['model']
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue
    c_deberta = '#ff7f00' # Orange for DeBERTa

    # Create color lists
    colors_0 = [c_deberta if 'deberta' in m else c_0 for m in models]
    colors_5 = [c_deberta if 'deberta' in m else c_5 for m in models]
    
    # Plot 0-shot
    rects1 = ax.barh(x - width/2, plot_df['zero_accuracy'], width, label='0-shot', color=colors_0, alpha=0.9)
    
    # Plot 5-shot
    rects2 = ax.barh(x + width/2, plot_df['five_accuracy'], width, label='5-shot', color=colors_5, alpha=0.9)
    
    # Labels
    ax.bar_label(rects1, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['zero_accuracy']], padding=3, fontsize=10)
    ax.bar_label(rects2, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['five_accuracy']], padding=3, fontsize=10)
    
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy: 0-shot vs 5-shot')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png')
    plt.close()
    print(f"Saved accuracy_comparison.png")

def plot_spearman(df: pd.DataFrame, output_dir: Path):
    """Plot Spearman correlation for all models (Zero-shot and Five-shot)."""
    cols = ['model', 'zero_spearman', 'five_spearman']
    plot_df = df[cols].copy()
    plot_df = plot_df.sort_values('zero_spearman', ascending=True)
    
    models = plot_df['model']
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue
    c_deberta = '#ff7f00' # Orange for DeBERTa

    # Create color lists
    colors_0 = [c_deberta if 'deberta' in m else c_0 for m in models]
    colors_5 = [c_deberta if 'deberta' in m else c_5 for m in models]
    
    # Plot 0-shot
    rects1 = ax.barh(x - width/2, plot_df['zero_spearman'], width, label='0-shot', color=colors_0, alpha=0.9)
    
    # Plot 5-shot
    rects2 = ax.barh(x + width/2, plot_df['five_spearman'], width, label='5-shot', color=colors_5, alpha=0.9)
    
    # Labels
    ax.bar_label(rects1, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['zero_spearman']], padding=3, fontsize=10)
    ax.bar_label(rects2, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['five_spearman']], padding=3, fontsize=10)
    
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlabel('Spearman Correlation')
    ax.set_title('Model Spearman Correlation: 0-shot vs 5-shot')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spearman_comparison.png')
    plt.close()
    print(f"Saved spearman_comparison.png")

def plot_time(df: pd.DataFrame, output_dir: Path):
    """Plot average inference time (Zero-shot and Five-shot)."""
    cols = ['model', 'zero_avg_time', 'five_avg_time']
    plot_df = df[cols].copy()
    
    # Remove models without time data (e.g. DeBERTa)
    plot_df = plot_df.dropna(subset=['zero_avg_time'])
    
    plot_df = plot_df.sort_values('zero_avg_time', ascending=True)
    
    if plot_df.empty:
        print("No time data available to plot.")
        return

    # Annotate model names
    annotated_models = []
    for m in plot_df['model']:
        name = m
        if 'think' in m:
            name += "\n(Thinking Model)"
        if 'gpt-oss-20b' in m:
            name += "\n(Colab/Different Hardware)"
        annotated_models.append(name)

    models = annotated_models
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue
    
    # Plot 0-shot
    rects1 = ax.barh(x - width/2, plot_df['zero_avg_time'], width, label='0-shot', color=c_0, alpha=0.9)
    
    # Plot 5-shot
    rects2 = ax.barh(x + width/2, plot_df['five_avg_time'], width, label='5-shot', color=c_5, alpha=0.9)
    
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlabel('Average Time per Sample (seconds)')
    ax.set_title('Inference Latency: 0-shot vs 5-shot')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    ax.bar_label(rects1, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['zero_avg_time']], padding=3, fontsize=10)
    ax.bar_label(rects2, labels=[f'{v:.2f}' if pd.notna(v) else '' for v in plot_df['five_avg_time']], padding=3, fontsize=10)
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time.png')
    plt.close()
    print(f"Saved inference_time.png")

def plot_improvement(df: pd.DataFrame, output_dir: Path):
    """Plot 0-shot vs 5-shot comparison for models that have both."""
    # Filter models that have both 0-shot and 5-shot accuracy
    plot_df = df.dropna(subset=['zero_accuracy', 'five_accuracy']).copy()
    
    if plot_df.empty:
        print("No models with both 0-shot and 5-shot data found.")
        return
        
    plot_df = plot_df.sort_values('five_accuracy', ascending=True)
    
    models = plot_df['model']
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue

    rects1 = ax.bar(x - width/2, plot_df['zero_accuracy'], width, label='0-shot', color=c_0, alpha=0.9)
    rects2 = ax.bar(x + width/2, plot_df['five_accuracy'], width, label='5-shot', color=c_5, alpha=0.9)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Few-shot Learning Impact: 0-shot vs 5-shot Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add improvement percentage annotation
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        impr = row['accuracy_impr_pct']
        if pd.notna(impr):
            height = max(row['zero_accuracy'], row['five_accuracy'])
            color = '#007000' if impr >= 0 else '#D00000' # Stronger Green if positive, Stronger Red if negative
            ax.text(i, height + 0.02, f'{impr:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_dir / 'few_shot_improvement.png')
    plt.close()
    print(f"Saved few_shot_improvement.png")

def plot_metric_consistency(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy vs Spearman correlation (Zero-shot and Five-shot) to check consistency."""
    # Prepare 0-shot data
    z_df = df[['model', 'zero_accuracy', 'zero_spearman']].dropna().copy()
    z_df = z_df.rename(columns={'zero_accuracy': 'accuracy', 'zero_spearman': 'spearman'})
    
    # Prepare 5-shot data
    f_df = df[['model', 'five_accuracy', 'five_spearman']].dropna().copy()
    f_df = f_df.rename(columns={'five_accuracy': 'accuracy', 'five_spearman': 'spearman'})
    
    if z_df.empty and f_df.empty:
        print("No data available for metric consistency plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue
    c_deberta = '#ff7f00' # Orange for DeBERTa

    # Plot 0-shot
    colors_z = [c_deberta if 'deberta' in m else c_0 for m in z_df['model']]
    ax.scatter(z_df['accuracy'], z_df['spearman'], color=colors_z, s=120, alpha=0.9, edgecolors='k', label='0-shot', zorder=3)
    
    # Plot 5-shot
    colors_f = [c_deberta if 'deberta' in m else c_5 for m in f_df['model']]
    ax.scatter(f_df['accuracy'], f_df['spearman'], color=colors_f, s=120, alpha=0.9, edgecolors='k', marker='s', label='5-shot', zorder=3)

    # Create custom legend to avoid DeBERTa color in 0-shot/5-shot labels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='0-shot', markerfacecolor=c_0, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label='5-shot', markerfacecolor=c_5, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='DeBERTa', markerfacecolor=c_deberta, markersize=10, markeredgecolor='k'),
    ]
    ax.legend(handles=legend_elements)

    # Connect points for same model
    common_models = set(z_df['model']).intersection(set(f_df['model']))
    for model in common_models:
        z_row = z_df[z_df['model'] == model].iloc[0]
        f_row = f_df[f_df['model'] == model].iloc[0]
        
        # Draw arrow from 0-shot to 5-shot
        ax.annotate("", 
                    xy=(f_row['accuracy'], f_row['spearman']), 
                    xytext=(z_row['accuracy'], z_row['spearman']),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=20, color="gray", alpha=0.7, lw=2),
                    zorder=2)

    # Add labels
    texts = []
    # Label 0-shot points only
    for _, row in z_df.iterrows():
        texts.append(ax.text(row['accuracy'], row['spearman'], row['model'], fontsize=9))

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Metric Consistency: Accuracy vs Spearman (0-shot & 5-shot)')
    # ax.legend() # Removed default legend
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_consistency.png')
    plt.close()
    print(f"Saved metric_consistency.png")

def plot_learning_potential(df: pd.DataFrame, output_dir: Path):
    """Plot Zero-shot Accuracy vs Accuracy Improvement %."""
    plot_df = df[['model', 'zero_accuracy', 'five_accuracy', 'accuracy_impr_pct']].dropna().copy()
    
    if plot_df.empty:
        print("No data available for learning potential plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    # Color points based on improvement (positive=green, negative=red)
    colors = []
    for idx, row in plot_df.iterrows():
        if 'deberta' in row['model']:
            colors.append('#ff7f00') # Orange for DeBERTa
        elif row['accuracy_impr_pct'] >= 0:
            colors.append('#55a868') # Green
        else:
            colors.append('#c44e52') # Red
    
    ax.scatter(plot_df['zero_accuracy'], plot_df['accuracy_impr_pct'], c=colors, s=100, alpha=0.8, edgecolors='w', zorder=3)
    
    # Draw arrows from 0-shot to 5-shot
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.annotate("", 
                    xy=(row['five_accuracy'], row['accuracy_impr_pct']), 
                    xytext=(row['zero_accuracy'], row['accuracy_impr_pct']),
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], alpha=0.6, lw=1.5),
                    zorder=2)
    
    # Add labels
    texts = []
    for _, row in plot_df.iterrows():
        # Offset label slightly up to avoid overlap with arrow/point
        texts.append(ax.text(row['zero_accuracy'], row['accuracy_impr_pct'] + 0.8, row['model'], fontsize=9, ha='center', va='bottom'))
        
    # Set x-axis limits to include both zero and five shot scores AND DeBERTa
    all_scores = pd.concat([plot_df['zero_accuracy'], plot_df['five_accuracy']])
    
    # Check for DeBERTa to include in limits
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
         val = deberta_row.iloc[0]['zero_accuracy']
         if pd.notna(val):
             all_scores = pd.concat([all_scores, pd.Series([val])])

    x_min, x_max = all_scores.min(), all_scores.max()
    padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_xlabel('Baseline Performance (Zero-shot Accuracy)')
    ax.set_ylabel('Benefit from Few-shot (Accuracy Improvement %)')
    ax.set_title('Learning Potential: Baseline vs Improvement')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # Zero improvement line
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add DeBERTa vertical line
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
        deberta_val = deberta_row.iloc[0]['zero_accuracy']
        if pd.notna(deberta_val):
            ax.axvline(deberta_val, color='#ff7f00', linestyle='--', linewidth=2, alpha=0.8, label='DeBERTa')
            # Add text label near the top of the line
            ylim = ax.get_ylim()
            ax.text(deberta_val, ylim[1] - (ylim[1]-ylim[0])*0.05, ' DeBERTa', color='#ff7f00', fontweight='bold', ha='left', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_potential.png')
    plt.close()
    print(f"Saved learning_potential.png")

def plot_learning_potential_spearman(df: pd.DataFrame, output_dir: Path):
    """Plot Zero-shot Spearman vs Spearman Improvement %."""
    plot_df = df[['model', 'zero_spearman', 'five_spearman', 'spearman_impr_pct']].dropna().copy()
    
    if plot_df.empty:
        print("No data available for learning potential (Spearman) plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    # Color points based on improvement (positive=green, negative=red), Orange for DeBERTa
    colors = []
    for idx, row in plot_df.iterrows():
        if 'deberta' in row['model']:
            colors.append('#ff7f00') # Orange for DeBERTa
        elif row['spearman_impr_pct'] >= 0:
            colors.append('#55a868') # Green
        else:
            colors.append('#c44e52') # Red
    
    ax.scatter(plot_df['zero_spearman'], plot_df['spearman_impr_pct'], c=colors, s=100, alpha=0.8, edgecolors='w', zorder=3)
    
    # Draw arrows from 0-shot to 5-shot
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.annotate("", 
                    xy=(row['five_spearman'], row['spearman_impr_pct']), 
                    xytext=(row['zero_spearman'], row['spearman_impr_pct']),
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], alpha=0.6, lw=1.5),
                    zorder=2)
    
    # Add labels
    texts = []
    for _, row in plot_df.iterrows():
        # Offset label slightly up to avoid overlap with arrow/point
        texts.append(ax.text(row['zero_spearman'], row['spearman_impr_pct'] + 0.8, row['model'], fontsize=9, ha='center', va='bottom'))
        
    # Set x-axis limits to include both zero and five shot scores AND DeBERTa
    all_scores = pd.concat([plot_df['zero_spearman'], plot_df['five_spearman']])
    
    # Check for DeBERTa to include in limits
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
         val = deberta_row.iloc[0]['zero_spearman']
         if pd.notna(val):
             all_scores = pd.concat([all_scores, pd.Series([val])])

    x_min, x_max = all_scores.min(), all_scores.max()
    padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_xlabel('Baseline Performance (Zero-shot Spearman)')
    ax.set_ylabel('Benefit from Few-shot (Spearman Improvement %)')
    ax.set_title('Learning Potential: Baseline vs Improvement (Spearman)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # Zero improvement line
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add DeBERTa vertical line
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
        deberta_val = deberta_row.iloc[0]['zero_spearman']
        if pd.notna(deberta_val):
            ax.axvline(deberta_val, color='#ff7f00', linestyle='--', linewidth=2, alpha=0.8, label='DeBERTa')
            # Add text label near the top of the line
            ylim = ax.get_ylim()
            ax.text(deberta_val, ylim[1] - (ylim[1]-ylim[0])*0.05, ' DeBERTa', color='#ff7f00', fontweight='bold', ha='left', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_potential_spearman.png')
    plt.close()
    print(f"Saved learning_potential_spearman.png")

def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    results_dir = repo_root / "results"
    csv_path = results_dir / "summary_0shot_5shot_scores.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    set_style()
    
    plot_accuracy(df, results_dir)
    plot_spearman(df, results_dir)
    plot_time(df, results_dir)
    plot_improvement(df, results_dir)
    plot_metric_consistency(df, results_dir)
    plot_learning_potential(df, results_dir)
    plot_learning_potential_spearman(df, results_dir)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
