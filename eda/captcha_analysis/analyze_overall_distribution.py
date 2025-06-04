import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def analyze_overall_distribution(dataset_paths):
    """
    Analyze the overall character and position distribution across all captcha types.
    
    Args:
        dataset_paths (list): List of paths to the dataset directories
    
    Returns:
        tuple: (char_distribution, position_distribution)
    """
    # Initialize counters
    char_distribution = defaultdict(int)
    position_distribution = defaultdict(lambda: defaultdict(int))
    
    # Walk through each dataset directory
    for dataset_path in dataset_paths:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Extract captcha text from filename
                        captcha_text = os.path.splitext(file)[0]
                        
                        # Update character distribution
                        for char in captcha_text:
                            char_distribution[char] += 1
                        
                        # Update position distribution
                        for pos, char in enumerate(captcha_text):
                            position_distribution[pos][char] += 1
                            
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
    
    return dict(char_distribution), dict(position_distribution)

def plot_overall_char_distribution(char_distribution, output_path):
    """
    Create a bar plot of overall character usage frequency.
    
    Args:
        char_distribution (dict): Dictionary containing character counts
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Sort characters alphabetically
    chars = sorted(char_distribution.keys())
    counts = [char_distribution[char] for char in chars]
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    
    # Create bar plot
    plt.bar(chars, percentages)
    plt.title('Overall Character Distribution')
    plt.xlabel('Character')
    plt.ylabel('Percentage')
    
    # Add percentage labels
    for i, v in enumerate(percentages):
        if v > 0:  # Only label non-zero values
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_overall_position_distribution(position_distribution, output_path):
    """
    Create a heatmap showing overall character distribution by position.
    
    Args:
        position_distribution (dict): Dictionary containing position-wise character counts
        output_path (str): Path to save the plot
    """
    # Get all unique characters and sort them alphabetically
    all_chars = sorted(set(char 
                         for pos_dist in position_distribution.values() 
                         for char in pos_dist.keys()))
    
    # Create matrix for heatmap
    matrix = np.zeros((len(all_chars), max(position_distribution.keys()) + 1))
    
    # Fill matrix with percentages
    for pos, char_counts in position_distribution.items():
        total = sum(char_counts.values())
        if total > 0:  # Avoid division by zero
            for char_idx, char in enumerate(all_chars):
                matrix[char_idx, pos] = char_counts.get(char, 0) / total * 100
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create heatmap
    im = plt.imshow(matrix, aspect='auto', cmap='YlOrRd')
    plt.title('Overall Character Distribution by Position')
    
    # Set labels
    plt.yticks(range(len(all_chars)), all_chars)
    plt.xticks(range(matrix.shape[1]), [f'Pos {i+1}' for i in range(matrix.shape[1])])
    
    # Add colorbar
    plt.colorbar(im, label='Percentage')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Define paths to include all dataset directories
    dataset_paths = [
        "data/train2",
        "data/val2",
        "data/test2"
    ]
    output_dir = Path("eda/captcha_analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze distributions
    print("Analyzing overall distributions...")
    char_distribution, position_distribution = analyze_overall_distribution(dataset_paths)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_overall_char_distribution(
        char_distribution,
        output_dir / "overall_char_distribution.png"
    )
    
    plot_overall_position_distribution(
        position_distribution,
        output_dir / "overall_position_distribution.png"
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved in: {output_dir}")
    
    # Print character distribution summary
    print("\nCharacter distribution:")
    total = sum(char_distribution.values())
    for char, count in sorted(char_distribution.items()):
        percentage = (count / total) * 100
        print(f"- {char}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main() 