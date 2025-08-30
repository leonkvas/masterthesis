import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define base directory
base_dir = r'C:\Users\Administrator\Cursor\masterthesis\multi-label-classification\Adversarial\20250608_163606\transferability_results'

# Create new directory for ordered plots
os.makedirs(os.path.join(base_dir, 'ordered_plots'), exist_ok=True)

# Read the CSV file
df = pd.read_csv(os.path.join(base_dir, 'transferability_results.csv'))

# Define the desired order for attack methods and models
attack_order = ['gaussian', 'fgsm', 'bim', 'ifgs', 'smooth_fgsm']
model_order = [
    'best_double_conv_layers_model',
    'best_enhanced_augmentation',
    'best_with_residual_connection',
   # 'best_robust_model',
    'best_robust_model2',
    'best_robust_augmented_model',
    #'best_robust_augmented_model2'
]

# Create a mapping for better model names in plots
model_names = {
    'best_double_conv_layers_model': 'Base DoubleConv',
    'best_enhanced_augmentation': 'Base Enhanced',
    'best_with_residual_connection': 'Base ResidualConn',
    'best_robust_model2': 'Robust',
    'best_robust_augmented_model': 'RobustAugmented'
}

# Define parameter labels for each attack method
param_labels = {
    'gaussian': 'Noise Std',
    'fgsm': 'Epsilon',
    'bim': 'Epsilon',
    'ifgs': 'Confidence',
    'smooth_fgsm': 'Sigma'
}

# Set the style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Function to create and save plots
def create_plots(df, metric, title, filename):
    plt.figure(figsize=(15, 8))
    
    # Create a pivot table with the desired ordering
    pivot_data = df.pivot_table(
        values=metric,
        index='model',
        columns='attack_method',
        aggfunc='mean'
    )
    
    # Reorder columns and rows
    pivot_data = pivot_data.reindex(columns=attack_order)
    pivot_data = pivot_data.reindex(index=model_order)
    
    # Create the heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': metric.replace('_', ' ').title()},
        xticklabels=[method.upper() for method in attack_order],
        yticklabels=[model_names[model] for model in model_order]
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'ordered_plots', f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create plots for both metrics
create_plots(
    df,
    'char_accuracy',
    'Average Character Accuracy Across Attack Methods',
    'average_char_accuracy'
)

create_plots(
    df,
    'success_rate',
    'Average FSA Across Attack Methods',
    'average_success_rate'
)

# Create individual plots for each attack method
for attack in attack_order:
    attack_df = df[df['attack_method'] == attack]
    
    if attack == 'smooth_fgsm':
        # Special handling for Smooth FGSM with two parameters
        # Extract sigma and epsilon values
        attack_df['sigma'] = attack_df['attack_parameter'].str.extract(r'sigma=([\d.]+)').astype(float)
        attack_df['epsilon'] = attack_df['attack_parameter'].str.extract(r'epsilon=([\d.]+)').astype(float)
        
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{attack.upper()} Attack Results', fontsize=16, y=1.02)
        
        # Get unique combinations of sigma and epsilon
        param_combinations = attack_df[['sigma', 'epsilon']].drop_duplicates().sort_values(['sigma', 'epsilon'])
        
        # Plot character accuracy
        for sigma, epsilon in param_combinations.itertuples(index=False):
            param_df = attack_df[(attack_df['sigma'] == sigma) & (attack_df['epsilon'] == epsilon)]
            data = param_df.groupby('model')['char_accuracy'].mean().reindex(model_order)
            ax1.plot(range(len(data)), data.values, 'o-', label=f'σ={sigma}, ε={epsilon}')
        
        ax1.set_xticks(range(len(model_order)))
        ax1.set_xticklabels([model_names[model] for model in model_order], rotation=45)
        ax1.set_title('Character Accuracy')
        ax1.set_ylim(0, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot success rate
        for sigma, epsilon in param_combinations.itertuples(index=False):
            param_df = attack_df[(attack_df['sigma'] == sigma) & (attack_df['epsilon'] == epsilon)]
            data = param_df.groupby('model')['success_rate'].mean().reindex(model_order)
            ax2.plot(range(len(data)), data.values, 'o-', label=f'σ={sigma}, ε={epsilon}')
        
        ax2.set_xticks(range(len(model_order)))
        ax2.set_xticklabels([model_names[model] for model in model_order], rotation=45)
        ax2.set_title('FSA')
        ax2.set_ylim(0, 1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    else:
        # Get unique parameter values for other attacks
        param_values = sorted(attack_df['attack_parameter'].unique())
        
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{attack.upper()} Attack Results', fontsize=16, y=1.02)
        
        # Plot character accuracy
        for param in param_values:
            param_df = attack_df[attack_df['attack_parameter'] == param]
            data = param_df.groupby('model')['char_accuracy'].mean().reindex(model_order)
            ax1.plot(range(len(data)), data.values, 'o-', label=f'{param_labels[attack]}={param}')
        
        ax1.set_xticks(range(len(model_order)))
        ax1.set_xticklabels([model_names[model] for model in model_order], rotation=45)
        ax1.set_title('Character Accuracy')
        ax1.set_ylim(0, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot success rate
        for param in param_values:
            param_df = attack_df[attack_df['attack_parameter'] == param]
            data = param_df.groupby('model')['success_rate'].mean().reindex(model_order)
            ax2.plot(range(len(data)), data.values, 'o-', label=f'{param_labels[attack]}={param}')
        
        ax2.set_xticks(range(len(model_order)))
        ax2.set_xticklabels([model_names[model] for model in model_order], rotation=45)
        ax2.set_title('FSA')
        ax2.set_ylim(0, 1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'ordered_plots', f'{attack}_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("Plots have been generated successfully in the 'ordered_plots' subdirectory!") 