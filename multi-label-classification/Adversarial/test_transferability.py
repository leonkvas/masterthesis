import tensorflow as tf
import numpy as np
import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Parameters (must match training) ---
IMG_SIZE = (50, 250)  # (height, width)
CHANNELS = 1  # Grayscale images

# Model paths
SOURCE_MODEL_PATH = "best_double_conv_layers_model.keras"
TARGET_MODEL_PATHS = [
    "best_with_residual_connection.keras",
    "best_enhanced_augmentation.keras",
    "best_with_residual_connection.keras"
]

# Adversarial examples directory
ADV_EXAMPLES_DIR = "multi-label-classification/Adversarial/transferability_examples"
RESULTS_DIR = "multi-label-classification/Adversarial/transferability_results"

# Vocabulary settings
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
MAX_CAPTCHA_LEN = 7

# Custom metric function that needs to be defined when loading models
@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    """Custom metric to measure full sequence accuracy."""
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# --- Helper functions ---
def load_and_preprocess_image(image_path):
    """Reads an image from 'image_path', resizes it to IMG_SIZE, and normalizes pixel values."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=CHANNELS)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0, 1]
    return img

def evaluate_transferability(model, model_name, metadata):
    """
    Evaluate transferability of adversarial examples on a target model.
    
    Args:
        model: The target model to evaluate
        model_name: Name of the model (for reporting)
        metadata: List of dictionaries containing metadata about adversarial examples
        
    Returns:
        Dictionary with transferability results
    """
    results = {}
    
    # Group examples by attack method and parameter
    attack_types = {}
    for example in metadata:
        attack_method = example["attack_method"]
        
        if attack_method not in attack_types:
            attack_types[attack_method] = []
        
        attack_types[attack_method].append(example)
    
    # Process each attack type
    for attack_method, examples in attack_types.items():
        print(f"\nEvaluating {attack_method} examples on {model_name}...")
        
        if attack_method not in results:
            results[attack_method] = []
        
        # Group by attack parameter
        param_groups = {}
        for ex in examples:
            param = ex.get("attack_parameter")
            
            # Handle special case for smooth_fgsm with multiple parameters
            if attack_method == "smooth_fgsm":
                params = ex.get("attack_parameters", {})
                sigma = params.get("sigma")
                epsilon = params.get("epsilon")
                param = f"sigma={sigma}_epsilon={epsilon}"
            
            if param not in param_groups:
                param_groups[param] = []
            
            param_groups[param].append(ex)
        
        # Evaluate each parameter group
        for param, group in param_groups.items():
            print(f"  Processing {attack_method} with parameter {param}, {len(group)} examples")
            
            correct_predictions = 0
            incorrect_predictions = 0
            
            # Track success of adversarial examples that worked on source model
            source_succeeded = [ex for ex in group if ex["attack_success"]]
            target_succeeded_from_source = 0
            
            for example in group:
                # Load adversarial image
                adv_path = os.path.join(ADV_EXAMPLES_DIR, example["relative_path"])
                adv_image = load_and_preprocess_image(adv_path)
                
                # Get original label
                original_label = example["original_label"]
                
                # Make prediction
                adv_img_batch = tf.expand_dims(adv_image, 0)
                preds = model.predict(adv_img_batch, verbose=0)
                pred_indices = np.argmax(preds, axis=-1)[0]
                prediction = ''.join([idx_to_char.get(idx, '') for idx in pred_indices if idx != 0])
                
                # Check if prediction is correct (matches original label)
                is_correct = prediction == original_label
                
                if is_correct:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                    # If this example succeeded on source model, count it
                    if example["attack_success"]:
                        target_succeeded_from_source += 1
            
            # Calculate transferability metrics
            total = len(group)
            success_rate = incorrect_predictions / total if total > 0 else 0
            
            # Calculate transfer ratio - how many examples that fooled source model also fool target
            transfer_ratio = 0
            if len(source_succeeded) > 0:
                transfer_ratio = target_succeeded_from_source / len(source_succeeded)
            
            result = {
                "model": model_name,
                "attack_method": attack_method,
                "attack_parameter": str(param),
                "total_examples": total,
                "successful_attacks": incorrect_predictions,
                "success_rate": success_rate,
                "source_model_successes": len(source_succeeded),
                "transferred_successes": target_succeeded_from_source,
                "transfer_ratio": transfer_ratio
            }
            
            results[attack_method].append(result)
            
            print(f"    Success rate: {success_rate:.2%} ({incorrect_predictions}/{total})")
            print(f"    Transfer ratio: {transfer_ratio:.2%} ({target_succeeded_from_source}/{len(source_succeeded)})")
    
    return results

def visualize_transferability(results, attack_methods):
    """Create visualizations for transferability results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare data for plotting
    for attack_method in attack_methods:
        all_results = []
        for model_results in results.values():
            if attack_method in model_results:
                all_results.extend(model_results[attack_method])
        
        # Skip if no results for this attack method
        if not all_results:
            continue
        
        df = pd.DataFrame(all_results)
        
        # Plot success rates
        plt.figure(figsize=(12, 8))
        sns.barplot(x='attack_parameter', y='success_rate', hue='model', data=df)
        plt.title(f'{attack_method.upper()} Attack Success Rate by Model')
        plt.xlabel('Attack Parameter')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{attack_method}_success_rates.png'), dpi=300)
        plt.close()
        
        # Plot transfer ratios
        plt.figure(figsize=(12, 8))
        sns.barplot(x='attack_parameter', y='transfer_ratio', hue='model', data=df)
        plt.title(f'{attack_method.upper()} Transfer Ratio by Model')
        plt.xlabel('Attack Parameter')
        plt.ylabel('Transfer Ratio (% of source successes that transferred)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{attack_method}_transfer_ratios.png'), dpi=300)
        plt.close()
    
    # Create comparison plot across all attack methods
    all_data = []
    for model_name, model_results in results.items():
        for attack_method, method_results in model_results.items():
            for result in method_results:
                result_copy = result.copy()
                result_copy["model"] = model_name
                all_data.append(result_copy)
    
    if all_data:
        df_all = pd.DataFrame(all_data)
        
        # Calculate average transfer ratio by attack method and model
        avg_transfer = df_all.groupby(['attack_method', 'model'])['transfer_ratio'].mean().reset_index()
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='attack_method', y='transfer_ratio', hue='model', data=avg_transfer)
        plt.title('Average Transfer Ratio by Attack Method and Model')
        plt.xlabel('Attack Method')
        plt.ylabel('Average Transfer Ratio')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'average_transfer_ratios.png'), dpi=300)
        plt.close()

def load_model_with_custom_objects(model_path):
    """Load model with custom metrics properly registered."""
    try:
        # Define custom objects dictionary for loading
        custom_objects = {
            'full_sequence_accuracy': full_sequence_accuracy
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model.trainable = False
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise e

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load metadata of generated adversarial examples
    metadata_path = os.path.join(ADV_EXAMPLES_DIR, "examples_metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata)} adversarial examples")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please run generate_adversarial_examples.py first to create adversarial examples.")
        return
    
    # Get attack methods from metadata
    attack_methods = set(ex["attack_method"] for ex in metadata)
    print(f"Found {len(attack_methods)} attack methods: {', '.join(attack_methods)}")
    
    # Load source model (for reference)
    print(f"Loading source model: {SOURCE_MODEL_PATH}")
    source_model = load_model_with_custom_objects(SOURCE_MODEL_PATH)
    
    # Process target models and gather results
    results = {}
    source_model_name = os.path.splitext(os.path.basename(SOURCE_MODEL_PATH))[0]
    
    # Test adversarial examples on source model (baseline)
    print(f"\nEvaluating on source model: {source_model_name}")
    source_results = evaluate_transferability(source_model, source_model_name, metadata)
    results[source_model_name] = source_results
    
    # Test transferability to target models
    for model_path in TARGET_MODEL_PATHS:
        try:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            print(f"\nLoading target model: {model_path}")
            
            model = load_model_with_custom_objects(model_path)
            
            print(f"Evaluating transferability on {model_name}")
            model_results = evaluate_transferability(model, model_name, metadata)
            results[model_name] = model_results
            
        except Exception as e:
            print(f"Error loading or evaluating model {model_path}: {e}")
    
    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, "transferability_results.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["model", "attack_method", "attack_parameter", "total_examples", 
                     "successful_attacks", "success_rate", "source_model_successes", 
                     "transferred_successes", "transfer_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model_name, model_results in results.items():
            for attack_method, method_results in model_results.items():
                for result in method_results:
                    writer.writerow(result)
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_transferability(results, attack_methods)
    
    print(f"\nTransferability testing complete!")
    print(f"Results saved to {csv_path}")
    print(f"Visualizations saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main() 