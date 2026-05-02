import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ite_distribution(ite, save_path):
    plt.figure()
    plt.hist(ite, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(ite.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ite.mean():.2f}')
    plt.xlabel("Estimated ITE")
    plt.ylabel("Count")
    plt.title("Distribution of Estimated Treatment Effects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ite_scatter(true_ite, pred_ite, save_path):
    plt.figure()
    plt.scatter(true_ite, pred_ite, alpha=0.3)
    # Add diagonal reference line
    min_val = min(true_ite.min(), pred_ite.min())
    max_val = max(true_ite.max(), pred_ite.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')
    plt.xlabel("True ITE")
    plt.ylabel("Estimated ITE")
    plt.title("True vs Estimated ITE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_policy_comparison(values_dict, save_path):
    plt.figure()
    bars = plt.bar(values_dict.keys(), values_dict.values())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Expected Outcome")
    plt.title("Policy Comparison")
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_budget_curve(budget_curve_df, save_path):
    plt.figure()
    treatment_percent = budget_curve_df["Treatment Rate"] * 100
    plt.plot(
        treatment_percent,
        budget_curve_df["Policy Value"],
        marker="o",
        linewidth=2,
        color="seagreen"
    )
    plt.xlabel("Treatment Budget (%)")
    plt.ylabel("Policy Value")
    plt.title("Policy Value by Treatment Budget")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(comparison_df, save_path, true_ate=None):
    """
    Plot comparison of multiple DML estimators.
    
    Args:
        comparison_df: DataFrame with columns ['Model', 'Estimated ATE', 'ATE Error', 'PEHE']
        save_path: Path to save the figure
        true_ate: Optional true ATE reference line for the evaluated split
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = comparison_df['Model'].values
    
    # Plot ATE Error
    ax1 = axes[0]
    bars1 = ax1.bar(models, comparison_df['ATE Error'].values, color='steelblue', edgecolor='black')
    ax1.set_ylabel('ATE Error')
    ax1.set_title('Average Treatment Effect Error')
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot PEHE
    ax2 = axes[1]
    bars2 = ax2.bar(models, comparison_df['PEHE'].values, color='coral', edgecolor='black')
    ax2.set_ylabel('PEHE')
    ax2.set_title('Precision in Estimation of Heterogeneous Effect')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot Estimated ATE vs reference ATE
    ax3 = axes[2]
    ax3.bar(models, comparison_df['Estimated ATE'].values, color='seagreen', edgecolor='black', alpha=0.7, label='Estimated')
    if true_ate is not None:
        ax3.axhline(y=true_ate, color='red', linestyle='--', linewidth=2, label='True ATE')
    ax3.set_ylabel('Estimated ATE')
    ax3.set_title('Estimated ATE by Model')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
