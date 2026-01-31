#!/usr/bin/env python3

import pandas as pd
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Add the scripts directory to path to allow importing plot_results
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plot_results
except ImportError:
    # If generic import fails (e.g. if script is run from root), try relative
    sys.path.append('scripts')
    import plot_results

def process_data(input_file):
    """Process the long-format CSV into the wide-format summary CSV."""
    df = pd.read_csv(input_file)
    
    # Dictionary to store processed rows: model_name -> dict of metrics
    processed_data = {}
    
    for _, row in df.iterrows():
        model_type = row['model_type']
        model_name = row['model_name']
        category = row['category']
        spearman = row['spearman']
        accuracy = row['accuracy']
        
        # Determine the target row name (model identifier) and metric prefix
        target_model_name = model_name
        metric_prefix = None # 'zero' or 'five'
        
        if model_type == 'DeBERTa':
            # DeBERTa models are treated as having "zero" metrics for plotting purposes
            # (or just simple values, but standardizing on zero_* allows reuse of plot code)
            target_model_name = model_name
            metric_prefix = 'zero'
        
        elif model_type == 'LLM':
            if category == 'zero-shot':
                target_model_name = model_name
                metric_prefix = 'zero'
            elif category == 'five-shot':
                target_model_name = model_name
                metric_prefix = 'five'
            elif category == 'zero-shot-deberta':
                target_model_name = f"{model_name}-deberta"
                metric_prefix = 'zero'
            elif category == 'five-shot-deberta':
                target_model_name = f"{model_name}-deberta"
                metric_prefix = 'five'
        
        if metric_prefix:
            if target_model_name not in processed_data:
                processed_data[target_model_name] = {'model': target_model_name}
            
            processed_data[target_model_name][f'{metric_prefix}_accuracy'] = accuracy
            processed_data[target_model_name][f'{metric_prefix}_spearman'] = spearman

    # Convert to DataFrame
    summary_df = pd.DataFrame(list(processed_data.values()))
    
    # Clean and normalize model names
    if hasattr(plot_results, 'clean_and_normalize_data'):
        summary_df = plot_results.clean_and_normalize_data(summary_df)
    
    # Calculate improvements
    if 'zero_accuracy' in summary_df.columns and 'five_accuracy' in summary_df.columns:
        summary_df['accuracy_impr_pct'] = (summary_df['five_accuracy'] - summary_df['zero_accuracy']) / summary_df['zero_accuracy'] * 100
    else:
        summary_df['accuracy_impr_pct'] = None
        
    if 'zero_spearman' in summary_df.columns and 'five_spearman' in summary_df.columns:
        summary_df['spearman_impr_pct'] = (summary_df['five_spearman'] - summary_df['zero_spearman']) / summary_df['zero_spearman'] * 100
    else:
        summary_df['spearman_impr_pct'] = None
        
    # Add empty time columns to match schema if necessary (though plot_results doesn't seem to strictly require them for plotting)
    summary_df['zero_avg_time'] = None
    summary_df['five_avg_time'] = None
    
    return summary_df

def plot_learning_potential_first100(df: pd.DataFrame, output_dir: Path):
    """Plot Zero-shot Accuracy vs Accuracy Improvement % (legend on left for first100 only)."""
    plot_df = df[['model', 'zero_accuracy', 'five_accuracy', 'accuracy_impr_pct']].dropna().copy()
    
    if plot_df.empty:
        print("No data available for learning potential plot.")
        return

    # Identify Standard vs Enhanced
    plot_df['base_name'] = plot_df['model'].apply(lambda x: x.replace('-deberta', ''))
    plot_df['is_enhanced'] = plot_df['model'].apply(lambda x: '-deberta' in x)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors
    c_std = '#1f78b4' # Blue (Standard)
    c_enh = '#6a3d9a' # Purple (Enhanced)
    c_orange = '#FFB74D' # Orange (Fine-tuned models)
    c_pos = '#55a868' # Green (Positive Improvement)
    c_neg = '#c44e52' # Red (Negative Improvement)
    
    # Plot points and 0->5 arrows
    for idx, row in plot_df.iterrows():
        # Determine point color
        if any(x in row['model'] for x in ['DeBERTa-NLI', 'deberta-base', 'bert-base', 'smollm']):
            color = c_orange
        elif row['is_enhanced']:
            color = c_enh
        else:
            color = c_std
            
        # Determine arrow color based on improvement
        impr = row['accuracy_impr_pct']
        arrow_color = c_pos if impr >= 0 else c_neg
        
        # Plot point
        ax.scatter(row['zero_accuracy'], row['accuracy_impr_pct'], color=color, s=120, alpha=0.9, edgecolors='w', zorder=3)
        
        # Draw arrow from 0-shot to 5-shot (X-axis shift)
        ax.annotate("", 
                    xy=(row['five_accuracy'], row['accuracy_impr_pct']), 
                    xytext=(row['zero_accuracy'], row['accuracy_impr_pct']),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, alpha=0.6, lw=2),
                    zorder=2)
        
        # Label point
        label = row['model'].replace('-deberta', '')
        if row['is_enhanced']:
            label += '*'
        ax.text(row['zero_accuracy'], row['accuracy_impr_pct'] + 0.8, label, fontsize=9, ha='center', va='bottom', alpha=0.8)

    # Connect Standard to Enhanced
    # Group by base_name
    for base_name, group in plot_df.groupby('base_name'):
        if len(group) == 2:
            # We have both Standard and Enhanced
            std = group[~group['is_enhanced']].iloc[0]
            enh = group[group['is_enhanced']].iloc[0]
            
            # Draw arrow from Standard to Enhanced
            ax.annotate("",
                        xy=(enh['zero_accuracy'], enh['accuracy_impr_pct']),
                        xytext=(std['zero_accuracy'], std['accuracy_impr_pct']),
                        arrowprops=dict(arrowstyle="->", color='gray', linestyle='--', alpha=0.5, lw=1.5),
                        zorder=1)

    # Set x-axis limits
    all_scores = pd.concat([plot_df['zero_accuracy'], plot_df['five_accuracy']])
    
    # Check for DeBERTa to include in limits
    deberta_row = df[df['model'] == 'DeBERTa-NLI']
        
    if not deberta_row.empty:
         val = deberta_row.iloc[0]['zero_accuracy']
         if pd.notna(val):
             all_scores = pd.concat([all_scores, pd.Series([val])])

    x_min, x_max = all_scores.min(), all_scores.max()
    padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_xlabel('Baseline Performance (Zero-shot Accuracy)')
    ax.set_ylabel('Benefit from Few-shot (Accuracy Improvement %)')
    ax.set_title('Few-shot Improvement vs Zero-shot Baseline (Accuracy)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # Zero improvement line
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add DeBERTa vertical line
    if not deberta_row.empty:
        deberta_val = deberta_row.iloc[0]['zero_accuracy']
        deberta_name = deberta_row.iloc[0]['model']
        label_text = ' DeBERTa-NLI'
        
        if pd.notna(deberta_val):
            ax.axvline(deberta_val, color=c_orange, linestyle='--', linewidth=2, alpha=0.8, label=label_text.strip())
            ylim = ax.get_ylim()
            ax.text(deberta_val, ylim[1] - (ylim[1]-ylim[0])*0.05, label_text, color=c_orange, fontweight='bold', ha='left', va='top', fontsize=9)
    
    # Custom Legend - MOVED TO LEFT for first100 plots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='LLM Only', markerfacecolor=c_std, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='LLM + DeBERTa', markerfacecolor=c_enh, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='DeBERTa-NLI', markerfacecolor=c_orange, markersize=10),
        Line2D([0], [0], color=c_pos, lw=2, label='Positive Improvement'),
        Line2D([0], [0], color=c_neg, lw=2, label='Negative Improvement'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='LLM -> +DeBERTa Shift'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_potential.png')
    plt.close()
    print(f"Saved learning_potential.png")

def main():
    base_path = Path('/home/niccolo/Torino/LLM-SamEval-T5')
    results_dir = base_path / 'results'
    input_file = results_dir / 'model_comparison_first100.csv'
    output_csv = results_dir / 'summary_first100.csv'
    output_plots_dir = results_dir / 'plots_100'
    full_summary_file = results_dir / 'summary_0shot_5shot_scores.csv'
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Processing {input_file}...")
    summary_df = process_data(input_file)
    
    # Add bert-base and deberta-base from the full summary
    if full_summary_file.exists():
        full_summary = pd.read_csv(full_summary_file)
        for model_name in ['bert-base', 'deberta-base']:
            model_row = full_summary[full_summary['model'] == model_name]
            if not model_row.empty:
                row_dict = {'model': model_name}
                model_row = model_row.iloc[0]
                if pd.notna(model_row['zero_accuracy']):
                    row_dict['zero_accuracy'] = model_row['zero_accuracy']
                if pd.notna(model_row['zero_spearman']):
                    row_dict['zero_spearman'] = model_row['zero_spearman']
                summary_df = pd.concat([summary_df, pd.DataFrame([row_dict])], ignore_index=True)
    
    print(f"Saving summary to {output_csv}...")
    summary_df.to_csv(output_csv, index=False)
    
    # Create output directory for plots
    output_plots_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots in {output_plots_dir}...")
    
    # Set style
    plot_results.set_style()
    
    # Generate plots using imported functions
    try:
        plot_results.plot_accuracy(summary_df, output_plots_dir)
        plot_results.plot_spearman(summary_df, output_plots_dir)
        plot_results.plot_improvement(summary_df, output_plots_dir)
        plot_results.plot_metric_consistency_superposed(summary_df, output_plots_dir)
        plot_learning_potential_first100(summary_df, output_plots_dir)  # Use custom function with left legend
        # Note: plot_learning_potential_spearman might crash if missing columns, let's check
        if hasattr(plot_results, 'plot_learning_potential_spearman'):
             plot_results.plot_learning_potential_spearman(summary_df, output_plots_dir)
             
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
