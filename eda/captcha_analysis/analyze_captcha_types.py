import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def rename_image_with_corrections(image_path, original_text, replacements):
    """
    Rename an image file with corrected text.
    
    Args:
        image_path (str): Path to the image file
        original_text (str): Original text from filename
        replacements (list): List of replacement operations
    
    Returns:
        str: New path if renamed, original path if no change needed
    """
    try:
        # Create corrected text
        corrected_text = list(original_text)
        for rep in replacements:
            corrected_text[rep['position'] - 1] = rep['replacement']
        corrected_text = ''.join(corrected_text)
        
        # If no change, return original path
        if corrected_text == original_text:
            return image_path
            
        # Create new path with corrected text
        dir_path = os.path.dirname(image_path)
        extension = os.path.splitext(image_path)[1]
        new_path = os.path.join(dir_path, f"{corrected_text}{extension}")
        
        # Rename the file
        os.rename(image_path, new_path)
        print(f"Renamed: {image_path} -> {new_path}")
        return new_path
        
    except Exception as e:
        print(f"Error renaming file {image_path}: {str(e)}")
        return image_path

def analyze_image_background(image_path):
    """
    Analyze the background characteristics of a captcha image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Dictionary containing background characteristics
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB for better color analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get average saturation and value
    avg_saturation = np.mean(hsv[:,:,1])
    
    # Determine captcha type based on saturation
    # Based on the histogram and corrected mappings:
    # Peak 1 (around 25): checkered
    # Peak 2 (around 50): light blue
    # Peak 3 (around 200): dark_blue
    if avg_saturation < 35:  # First peak
        captcha_type = "checkered"  # Low saturation
    elif avg_saturation < 100:  # Second peak
        captcha_type = "light_blue"  # Medium saturation 
    else:  # Third peak (high saturation)
        captcha_type = "dark_blue"  # High saturation
    
    # Get edge density for additional analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (height * width)
    
    # Get dominant color
    pixels = img_rgb.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_color = unique_colors[np.argmax(counts)]
    
    return {
        "type": captcha_type,
        "edge_density": edge_density,
        "dominant_color": dominant_color.tolist(),
        "avg_saturation": avg_saturation,
        "avg_value": np.mean(hsv[:,:,2])
    }

def analyze_captcha_types(dataset_paths):
    """
    Analyze the distribution of captcha types in the dataset.
    
    Args:
        dataset_paths (list): List of paths to the dataset directories
    
    Returns:
        dict: Dictionary containing counts and characteristics for each captcha type
    """
    # Initialize counters and characteristics storage
    captcha_counts = defaultdict(int)
    type_characteristics = defaultdict(list)
    length_distribution = defaultdict(lambda: defaultdict(int))
    suspicious_files = []  # Track files with unexpected lengths
    suspicious_char_files = []  # Track files with suspicious characters
    non_numeric_position_files = []  # Track files with non-numeric characters in positions 1, 4, 7
    renamed_files = []  # Track files that were renamed
    
    # Initialize character distribution tracking
    char_distribution = defaultdict(lambda: defaultdict(int))  # type -> char -> count
    first_char_distribution = defaultdict(lambda: defaultdict(int))  # type -> first_char -> count
    position_char_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # type -> position -> char -> count
    
    SUSPICIOUS_CHARS = {'$', '&'}  # Define suspicious characters
    NUMBER_POSITIONS = {0, 3, 6}  # Positions (0-based) that should be numbers
    CHAR_REPLACEMENTS = {
        'I': '1',
        'S': '5',
        'O': '0'
    }
    
    # Walk through each dataset directory
    for dataset_path in dataset_paths:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image_path = os.path.join(root, file)
                        analysis = analyze_image_background(image_path)
                        
                        if analysis:
                            captcha_type = analysis["type"]
                            captcha_counts[captcha_type] += 1
                            type_characteristics[captcha_type].append(analysis)
                            
                            # Extract captcha text from filename
                            captcha_text = os.path.splitext(file)[0]
                            text_length = len(captcha_text)
                            
                            # Check for characters that need replacement ONLY in positions 1, 4, 7
                            replacements = []
                            for pos, char in enumerate(captcha_text):
                                if pos in NUMBER_POSITIONS and char in CHAR_REPLACEMENTS:
                                    replacements.append({
                                        'position': pos + 1,
                                        'original': char,
                                        'replacement': CHAR_REPLACEMENTS[char]
                                    })
                            
                            # If replacements needed, rename the file
                            if replacements:
                                old_path = image_path
                                new_path = rename_image_with_corrections(image_path, captcha_text, replacements)
                                
                                if old_path != new_path:
                                    renamed_files.append({
                                        'old_path': old_path,
                                        'new_path': new_path,
                                        'old_text': captcha_text,
                                        'new_text': os.path.splitext(os.path.basename(new_path))[0],
                                        'type': captcha_type,
                                        'replacements': replacements
                                    })
                                
                                # Update image_path and text for further processing
                                image_path = new_path
                                captcha_text = os.path.splitext(os.path.basename(new_path))[0]
                            
                            # Update length distribution with possibly corrected text
                            length_distribution[captcha_type][len(captcha_text)] += 1
                            
                            # Track character distributions with corrected text
                            for char in captcha_text:
                                char_distribution[captcha_type][char] += 1
                            
                            # Track first character distribution with corrected text
                            if captcha_text:
                                first_char_distribution[captcha_type][captcha_text[0]] += 1
                            
                            # Track character distribution by position with corrected text
                            for pos, char in enumerate(captcha_text):
                                position_char_distribution[captcha_type][pos][char] += 1
                                
                                # Check if position should be numeric but isn't
                                if pos in NUMBER_POSITIONS and not char.isdigit():
                                    # Don't report if it's a replaceable character in these positions
                                    if char not in CHAR_REPLACEMENTS:
                                        non_numeric_position_files.append({
                                            'path': image_path,
                                            'text': captcha_text,
                                            'type': captcha_type,
                                            'position': pos + 1,  # Convert to 1-based position
                                            'character': char
                                        })
                            
                            # Track suspicious characters
                            if any(char in SUSPICIOUS_CHARS for char in captcha_text):
                                suspicious_char_files.append({
                                    'path': image_path,
                                    'text': captcha_text,
                                    'type': captcha_type,
                                    'suspicious_chars': [char for char in captcha_text if char in SUSPICIOUS_CHARS]
                                })
                            
                            # Track suspicious lengths (not 6 or 7)
                            if text_length not in [6, 7]:
                                suspicious_files.append({
                                    'path': image_path,
                                    'length': text_length,
                                    'text': captcha_text,
                                    'type': captcha_type
                                })
                            
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
    
    # Print files that were renamed
    if renamed_files:
        print("\nFiles renamed with character corrections:")
        print("====================================")
        for file_info in renamed_files:
            print(f"Type: {file_info['type']}")
            print(f"Old path: {file_info['old_path']}")
            print(f"New path: {file_info['new_path']}")
            print(f"Old text: {file_info['old_text']}")
            print(f"New text: {file_info['new_text']}")
            print("Replacements made:")
            for rep in file_info['replacements']:
                print(f"  Position {rep['position']}: '{rep['original']}' -> '{rep['replacement']}'")
            print("--------------------------------")
    
    # Print suspicious files
    if suspicious_files:
        print("\nSuspicious file lengths detected:")
        print("================================")
        for file_info in suspicious_files:
            print(f"Path: {file_info['path']}")
            print(f"Type: {file_info['type']}")
            print(f"Text: {file_info['text']} (length: {file_info['length']})")
            print("--------------------------------")
    
    # Print files with suspicious characters
    if suspicious_char_files:
        print("\nSuspicious characters detected:")
        print("============================")
        for file_info in suspicious_char_files:
            print(f"Path: {file_info['path']}")
            print(f"Type: {file_info['type']}")
            print(f"Text: {file_info['text']}")
            print(f"Suspicious characters: {', '.join(file_info['suspicious_chars'])}")
            print("--------------------------------")
    
    # Print files with non-numeric characters in positions 1, 4, 7
    if non_numeric_position_files:
        print("\nNon-numeric characters in positions 1, 4, or 7 detected:")
        print("===================================================")
        for file_info in non_numeric_position_files:
            print(f"Path: {file_info['path']}")
            print(f"Type: {file_info['type']}")
            print(f"Text: {file_info['text']}")
            print(f"Position {file_info['position']}: '{file_info['character']}' (should be numeric)")
            print("--------------------------------")
    
    return (dict(captcha_counts), dict(type_characteristics), dict(length_distribution), 
            dict(char_distribution), dict(first_char_distribution), dict(position_char_distribution))

def plot_captcha_distribution(counts, output_path):
    """
    Create a bar plot of captcha type distribution.
    
    Args:
        counts (dict): Dictionary of captcha type counts
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values())
    plt.title('Distribution of Captcha Types in Dataset')
    plt.xlabel('Captcha Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts.values()):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig(output_path)
    plt.close()

def plot_type_characteristics(characteristics, output_path):
    """
    Create plots showing the characteristics of each captcha type.
    
    Args:
        characteristics (dict): Dictionary of characteristics for each type
        output_path (str): Path to save the plots
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Captcha Type Characteristics')
    
    # Plot edge density distribution
    for captcha_type, data in characteristics.items():
        edge_densities = [d['edge_density'] for d in data]
        axes[0, 0].hist(edge_densities, alpha=0.5, label=captcha_type)
    axes[0, 0].set_title('Edge Density Distribution')
    axes[0, 0].legend()
    
    # Plot average saturation
    for captcha_type, data in characteristics.items():
        saturations = [d['avg_saturation'] for d in data]
        axes[0, 1].hist(saturations, alpha=0.5, label=captcha_type)
    axes[0, 1].set_title('Saturation Distribution')
    axes[0, 1].legend()
    
    # Plot average value
    for captcha_type, data in characteristics.items():
        values = [d['avg_value'] for d in data]
        axes[1, 0].hist(values, alpha=0.5, label=captcha_type)
    axes[1, 0].set_title('Value (Brightness) Distribution')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_length_distribution(length_distribution, output_path):
    """
    Create a grouped bar plot showing the length distribution for each captcha type.
    
    Args:
        length_distribution (dict): Dictionary containing length distributions per type
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages and prepare data for plotting
    type_data = {}
    for captcha_type, lengths in length_distribution.items():
        total = sum(lengths.values())
        percentages = {length: (count/total)*100 for length, count in lengths.items()}
        type_data[captcha_type] = percentages
    
    # Get all unique lengths
    all_lengths = sorted(set(length for type_lengths in length_distribution.values() 
                           for length in type_lengths.keys()))
    
    # Set up the bar positions
    bar_width = 0.25
    r = np.arange(len(all_lengths))
    
    # Plot bars for each type
    for i, (captcha_type, percentages) in enumerate(sorted(type_data.items())):
        values = [percentages.get(length, 0) for length in all_lengths]
        plt.bar(r + i*bar_width, values, bar_width, label=captcha_type)
        
        # Add percentage labels on top of bars
        for j, value in enumerate(values):
            if value > 0:  # Only label non-zero values
                plt.text(r[j] + i*bar_width, value, f'{value:.1f}%', 
                        ha='center', va='bottom')
    
    plt.xlabel('Captcha Length')
    plt.ylabel('Percentage')
    plt.title('Captcha Length Distribution by Type')
    plt.xticks(r + bar_width, all_lengths)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(counts, characteristics, length_distribution, output_path):
    """
    Generate a text report of the analysis.
    
    Args:
        counts (dict): Dictionary of captcha type counts
        characteristics (dict): Dictionary of characteristics for each type
        length_distribution (dict): Dictionary of length distributions per type
        output_path (str): Path to save the report
    """
    total = sum(counts.values())
    
    with open(output_path, 'w') as f:
        f.write("Captcha Type Analysis Report\n")
        f.write("==========================\n\n")
        
        f.write(f"Total number of captchas: {total}\n\n")
        
        f.write("Distribution by type:\n")
        for captcha_type, count in counts.items():
            percentage = (count / total) * 100
            f.write(f"- {captcha_type}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nLength distribution by type:\n")
        for captcha_type, lengths in length_distribution.items():
            f.write(f"\n{captcha_type}:\n")
            type_total = sum(lengths.values())
            for length, count in sorted(lengths.items()):
                percentage = (count / type_total) * 100
                f.write(f"  - Length {length}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nCharacteristics by type:\n")
        for captcha_type, data in characteristics.items():
            f.write(f"\n{captcha_type}:\n")
            if data:
                avg_edge = np.mean([d['edge_density'] for d in data])
                avg_sat = np.mean([d['avg_saturation'] for d in data])
                avg_val = np.mean([d['avg_value'] for d in data])
                f.write(f"  - Average edge density: {avg_edge:.4f}\n")
                f.write(f"  - Average saturation: {avg_sat:.2f}\n")
                f.write(f"  - Average value: {avg_val:.2f}\n")

def plot_example_captchas(dataset_path, type_characteristics, output_path):
    """
    Plot one example image from each captcha type.
    
    Args:
        dataset_path (str): Path to the dataset directory
        type_characteristics (dict): Dictionary containing characteristics for each type
        output_path (str): Path to save the example plot
    """
    # Create a figure with subplots for each type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Example Captchas by Type')
    
    # Dictionary to store first found image path for each type
    examples = {}
    
    # Walk through the dataset to find examples
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                analysis = analyze_image_background(image_path)
                
                if analysis and analysis["type"] not in examples:
                    examples[analysis["type"]] = image_path
                    
                # If we have all types, stop searching
                if len(examples) == 3:
                    break
        if len(examples) == 3:
            break
    
    # Plot each example
    for idx, (captcha_type, image_path) in enumerate(sorted(examples.items())):
        # Read and display image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f'Type: {captcha_type}\nPath: {os.path.basename(image_path)}')
        axes[idx].axis('off')
        
        # Add saturation value as text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:,:,1])
        axes[idx].text(0.5, -0.1, f'Avg Saturation: {avg_saturation:.2f}', 
                      ha='center', transform=axes[idx].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_char_distribution(char_distribution, output_path):
    """
    Create a heatmap of character usage frequency for each captcha type.
    
    Args:
        char_distribution (dict): Dictionary containing character distributions per type
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Get all unique characters across all types
    all_chars = sorted(set(char 
                         for type_dist in char_distribution.values() 
                         for char in type_dist.keys()))
    
    # Create subplots for each captcha type
    num_types = len(char_distribution)
    fig, axes = plt.subplots(num_types, 1, figsize=(15, 5*num_types))
    if num_types == 1:
        axes = [axes]
    
    for idx, (captcha_type, char_counts) in enumerate(sorted(char_distribution.items())):
        # Calculate percentages
        total = sum(char_counts.values())
        percentages = [char_counts.get(char, 0)/total*100 for char in all_chars]
        
        # Create bar plot
        axes[idx].bar(all_chars, percentages)
        axes[idx].set_title(f'Character Distribution - {captcha_type}')
        axes[idx].set_xlabel('Character')
        axes[idx].set_ylabel('Percentage')
        
        # Add percentage labels
        for i, v in enumerate(percentages):
            if v > 0:  # Only label non-zero values
                axes[idx].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_position_distribution(position_char_distribution, output_path):
    """
    Create heatmaps showing character distribution by position for each type.
    
    Args:
        position_char_distribution (dict): Dictionary containing position-wise character distributions
        output_path (str): Path to save the plot
    """
    # Get all unique characters and sort them alphabetically
    all_chars = sorted(set(char 
                         for type_dist in position_char_distribution.values() 
                         for pos_dist in type_dist.values() 
                         for char in pos_dist.keys()))
    
    # Create subplot for each captcha type
    num_types = len(position_char_distribution)
    fig, axes = plt.subplots(num_types, 1, figsize=(15, 5*num_types))
    if num_types == 1:
        axes = [axes]
    
    for idx, (captcha_type, positions) in enumerate(sorted(position_char_distribution.items())):
        # Create matrix for heatmap
        matrix = np.zeros((len(all_chars), max(positions.keys()) + 1))
        
        # Fill matrix with percentages
        for pos, char_counts in positions.items():
            total = sum(char_counts.values())
            if total > 0:  # Avoid division by zero
                for char_idx, char in enumerate(all_chars):
                    matrix[char_idx, pos] = char_counts.get(char, 0) / total * 100
        
        # Create heatmap
        im = axes[idx].imshow(matrix, aspect='auto', cmap='YlOrRd')
        axes[idx].set_title(f'Character Distribution by Position - {captcha_type}')
        
        # Set labels with alphabetically sorted characters
        axes[idx].set_yticks(range(len(all_chars)))
        axes[idx].set_yticklabels(all_chars)
        axes[idx].set_xticks(range(matrix.shape[1]))
        axes[idx].set_xticklabels([f'Pos {i+1}' for i in range(matrix.shape[1])])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], label='Percentage')
    
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
    
    # Analyze captcha types
    print("Analyzing captcha types...")
    print(f"Processing directories: {', '.join(dataset_paths)}")
    (captcha_counts, type_characteristics, length_distribution, 
     char_distribution, first_char_distribution, position_char_distribution) = analyze_captcha_types(dataset_paths)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_captcha_distribution(
        captcha_counts,
        output_dir / "captcha_distribution.png"
    )
    
    plot_type_characteristics(
        type_characteristics,
        output_dir / "captcha_characteristics.png"
    )
    
    plot_length_distribution(
        length_distribution,
        output_dir / "captcha_length_distribution.png"
    )
    
    plot_char_distribution(
        char_distribution,
        output_dir / "char_distribution.png"
    )
    
    plot_position_distribution(
        position_char_distribution,
        output_dir / "position_distribution.png"
    )
    
    # Plot example captchas
    print("Generating example captchas plot...")
    plot_example_captchas(
        dataset_paths[0],  # Use train2 for examples
        type_characteristics,
        output_dir / "example_captchas.png"
    )
    
    # Generate report
    print("Generating report...")
    generate_report(
        captcha_counts,
        type_characteristics,
        length_distribution,
        output_dir / "captcha_analysis_report.txt"
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved in: {output_dir}")
    print("\nCaptcha type distribution:")
    for captcha_type, count in captcha_counts.items():
        print(f"- {captcha_type}: {count}")
    
    print("\nLength distribution by type:")
    for captcha_type, lengths in length_distribution.items():
        print(f"\n{captcha_type}:")
        type_total = sum(lengths.values())
        for length, count in sorted(lengths.items()):
            percentage = (count / type_total) * 100
            print(f"  - Length {length}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main() 