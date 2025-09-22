import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# --- Parameters (must match training) ---
IMG_SIZE = (50, 250)  # (height, width)
BATCH_SIZE = 16
NOISE_LEVELS = [0.1, 0.2, 0.25]  # Standard deviation of Gaussian noise
CHANNELS = 1  # Grayscale images

# Vocabulary settings (same as during training)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
MAX_CAPTCHA_LEN = 7


# --- Helper functions ---
def load_and_preprocess_image(image_path):
    """
    Reads an image from 'image_path', resizes it to IMG_SIZE, and normalizes pixel values.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=CHANNELS)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0, 1]
    return img


def extract_label_from_filename(image_path):
    """
    Extracts the ground-truth label (a string) from the image filename.
    Assumes the filename (without extension) is the CAPTCHA text.
    """
    base_name = os.path.basename(image_path)
    label_str = os.path.splitext(base_name)[0]
    return label_str


def label_to_sequence(label_str, max_len=MAX_CAPTCHA_LEN):
    """
    Converts a label string (e.g., "AB12") into a padded sequence of integers.
    """
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')[0]
    return np.array(seq_padded, dtype=np.int32)


def add_gaussian_noise(image, std_dev):
    """
    Add Gaussian noise to the image.
    
    Args:
        image: Normalized image tensor (values in [0,1])
        std_dev: Standard deviation of the Gaussian noise
        
    Returns:
        Noisy image tensor clipped to [0,1]
    """
    # Generate Gaussian noise with the same shape as the image
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std_dev, dtype=tf.float32)
    
    # Add noise to the image
    noisy_image = image + noise
    
    # Clip values to [0, 1] range
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    
    # Print noise statistics
    print(f"\nNoise Statistics (std_dev={std_dev}):")
    print(f"Min: {tf.reduce_min(noise).numpy()}")
    print(f"Max: {tf.reduce_max(noise).numpy()}")
    print(f"Mean: {tf.reduce_mean(noise).numpy()}")
    
    return noisy_image


def predict_captcha(model, image_path, noise_level):
    """
    Loads an image, adds Gaussian noise, and returns the prediction on the noisy image.
    """
    # Load and preprocess the original image
    img = load_and_preprocess_image(image_path)
    # Extract the ground truth label from filename
    label_str = extract_label_from_filename(image_path)
    true_label = label_str  # For printing purposes

    # Generate noisy image
    noisy_img = add_gaussian_noise(img, noise_level)

    # Perform prediction on the noisy image
    noisy_img_batch = tf.expand_dims(noisy_img, axis=0)
    preds = model.predict(noisy_img_batch, verbose=0)
    pred_indices = np.argmax(preds, axis=-1)[0]
    predicted_text = ''.join([idx_to_char.get(idx, '') for idx in pred_indices if idx != 0])

    return true_label, predicted_text, noisy_img


def visualize_attack(original_image, noisy_image, original_label, 
                    predicted_original, predicted_noisy, noise_level, save_path=None):
    """Visualize original and noisy images with predictions."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {original_label}\nPred: {predicted_original}')
    plt.axis('off')
    
    # Noisy image
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Gaussian Noise (σ={noise_level})\nPred: {predicted_noisy}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 3, 3)
    difference = np.abs(noisy_image.numpy().squeeze() - original_image.numpy().squeeze())
    plt.imshow(difference, cmap='hot', vmin=0, vmax=1)  # Fixed color scale
    plt.title('Noise Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_model_robustness(model, test_dir, num_samples=5):
    """Test model robustness against Gaussian noise."""
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    test_files = test_files[:num_samples]  # Limit number of samples
    
    results = {sigma: {'successful_attacks': 0, 'total': 0,
                       'original_loss': 0, 'noisy_loss': 0,
                       'original_accuracy': 0, 'noisy_accuracy': 0} 
               for sigma in NOISE_LEVELS}
    
    skipped_samples = 0
    processed_samples = 0
    
    for file in test_files:
        image_path = os.path.join(test_dir, file)
        original_image = load_and_preprocess_image(image_path)
        original_label = extract_label_from_filename(file)
        label_sequence = label_to_sequence(original_label)
        label_sequence_batch = tf.expand_dims(label_sequence, 0)
        
        # Get original prediction
        original_img_batch = tf.expand_dims(original_image, 0)
        original_preds = model.predict(original_img_batch, verbose=0)
        original_pred_indices = np.argmax(original_preds, axis=-1)[0]
        predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
        
        # Check if original prediction is correct, skip if not
        if predicted_original != original_label:
            print(f"Skipping {file}: original prediction '{predicted_original}' doesn't match label '{original_label}'")
            skipped_samples += 1
            continue
        
        processed_samples += 1
        
        # Calculate original loss
        original_loss = tf.keras.losses.sparse_categorical_crossentropy(
            label_sequence_batch, original_preds, from_logits=False
        )
        original_loss = tf.reduce_mean(original_loss).numpy()
        
        # Calculate character-level accuracy
        original_correct = tf.cast(tf.equal(tf.cast(label_sequence_batch, tf.int64), 
                                         tf.argmax(original_preds, axis=-1)), tf.float32)
        original_char_accuracy = tf.reduce_mean(original_correct).numpy()
        
        # Calculate sequence-level accuracy (full sequence correct)
        original_seq_correct = tf.reduce_all(tf.equal(tf.cast(label_sequence_batch, tf.int64),
                                                   tf.argmax(original_preds, axis=-1)), axis=1)
        original_seq_accuracy = tf.reduce_mean(tf.cast(original_seq_correct, tf.float32)).numpy()
        
        true_label = original_label  # For consistency with existing code
        
        for sigma in NOISE_LEVELS:
            # Create noisy example
            noisy_image = add_gaussian_noise(original_image, sigma)
            
            # Get noisy prediction
            noisy_img_batch = tf.expand_dims(noisy_image, axis=0)
            noisy_preds = model.predict(noisy_img_batch, verbose=0)
            noisy_pred_indices = np.argmax(noisy_preds, axis=-1)[0]
            predicted_noisy = ''.join([idx_to_char.get(idx, '') for idx in noisy_pred_indices if idx != 0])
            
            # Calculate noisy loss
            noisy_loss = tf.keras.losses.sparse_categorical_crossentropy(
                label_sequence_batch, noisy_preds, from_logits=False
            )
            noisy_loss = tf.reduce_mean(noisy_loss).numpy()
            
            # Calculate character-level accuracy
            noisy_correct = tf.cast(tf.equal(tf.cast(label_sequence_batch, tf.int64), 
                                          tf.argmax(noisy_preds, axis=-1)), tf.float32)
            noisy_char_accuracy = tf.reduce_mean(noisy_correct).numpy()
            
            # Calculate sequence-level accuracy (full sequence correct)
            noisy_seq_correct = tf.reduce_all(tf.equal(tf.cast(label_sequence_batch, tf.int64),
                                                   tf.argmax(noisy_preds, axis=-1)), axis=1)
            noisy_seq_accuracy = tf.reduce_mean(tf.cast(noisy_seq_correct, tf.float32)).numpy()
            
            # Check if attack was successful
            if predicted_noisy != true_label:
                results[sigma]['successful_attacks'] += 1
                
                # Save visualization for successful attacks
                save_path = f'multi-label-classification/Adversarial/noise_results/examples/sigma_{sigma}_{file}'
                visualize_attack(
                    original_image, noisy_image, original_label,
                    predicted_original, predicted_noisy, sigma, save_path
                )
            
            # Update metrics
            results[sigma]['total'] += 1
            results[sigma]['original_loss'] += original_loss
            results[sigma]['noisy_loss'] += noisy_loss
            results[sigma]['original_accuracy'] += original_seq_accuracy
            results[sigma]['noisy_accuracy'] += noisy_seq_accuracy
    
    # Calculate averages
    for sigma in NOISE_LEVELS:
        if results[sigma]['total'] > 0:
            results[sigma]['original_loss'] /= results[sigma]['total']
            results[sigma]['noisy_loss'] /= results[sigma]['total']
            results[sigma]['original_accuracy'] /= results[sigma]['total']
            results[sigma]['noisy_accuracy'] /= results[sigma]['total']
    
    # Print summary of processed vs skipped samples
    print(f"\nSummary of processed samples:")
    print(f"  Total images checked: {len(test_files)}")
    print(f"  Images with correct original predictions: {processed_samples}")
    print(f"  Images skipped (incorrect original predictions): {skipped_samples}")
    
    return results

@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def main():
    # Create results directory
    os.makedirs('multi-label-classification/Adversarial/noise_results', exist_ok=True)
    # Create examples subdirectory for images
    os.makedirs('multi-label-classification/Adversarial/noise_results/examples', exist_ok=True)
    
    # Load the trained model
    model = tf.keras.models.load_model("best_double_conv_layers_model.keras")
    
    # Set the model to evaluation mode
    model.trainable = False
    
    # Test directory
    test_dir = "data/test2"
    
    # Calculate max CAPTCHA length from test files
    test_files = [os.path.splitext(f)[0] for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    max_captcha_len = max(len(name) for name in test_files)
    print(f"Detected max CAPTCHA length: {max_captcha_len}")
    
    # Test model robustness
    print("\nTesting model robustness against Gaussian noise...")
    results = test_model_robustness(model, test_dir)
    
    # Print results
    print("\nAttack Success Rates:")
    for sigma in NOISE_LEVELS:
        success_rate = results[sigma]['successful_attacks'] / results[sigma]['total']
        print(f"σ = {sigma}: {success_rate:.2%} ({results[sigma]['successful_attacks']}/{results[sigma]['total']})")
    
    # Print accuracy and loss metrics
    print("\nAccuracy and Loss Metrics:")
    for sigma in NOISE_LEVELS:
        print(f"\nSigma = {sigma}:")
        print(f"  Original Accuracy: {results[sigma]['original_accuracy']:.4f}")
        print(f"  Noisy Accuracy: {results[sigma]['noisy_accuracy']:.4f}")
        print(f"  Accuracy Reduction: {results[sigma]['original_accuracy'] - results[sigma]['noisy_accuracy']:.4f}")
        print(f"  Original Loss: {results[sigma]['original_loss']:.4f}")
        print(f"  Noisy Loss: {results[sigma]['noisy_loss']:.4f}")
        print(f"  Loss Increase: {results[sigma]['noisy_loss'] - results[sigma]['original_loss']:.4f}")
    
    # Plot success rates
    plt.figure(figsize=(15, 5))
    
    # Plot attack success rates
    plt.subplot(1, 3, 1)
    noise_levels = list(results.keys())
    success_rates = [results[sigma]['successful_attacks'] / results[sigma]['total'] for sigma in noise_levels]
    
    plt.plot(noise_levels, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Attack Success Rate')
    plt.title('Gaussian Noise Attack Success Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot accuracy comparison
    plt.subplot(1, 3, 2)
    original_accuracies = [results[sigma]['original_accuracy'] for sigma in noise_levels]
    noisy_accuracies = [results[sigma]['noisy_accuracy'] for sigma in noise_levels]
    
    plt.plot(noise_levels, original_accuracies, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(noise_levels, noisy_accuracies, 'o-', linewidth=2, markersize=8, label='Noisy')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot loss comparison
    plt.subplot(1, 3, 3)
    original_losses = [results[sigma]['original_loss'] for sigma in noise_levels]
    noisy_losses = [results[sigma]['noisy_loss'] for sigma in noise_levels]
    
    plt.plot(noise_levels, original_losses, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(noise_levels, noisy_losses, 'o-', linewidth=2, markersize=8, label='Noisy')
    plt.xlabel('Sigma (Noise Level)')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/noise_results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    import csv
    with open('multi-label-classification/Adversarial/noise_results/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Sigma', 'Success_Rate', 'Original_Accuracy', 'Noisy_Accuracy', 
                     'Accuracy_Reduction', 'Original_Loss', 'Noisy_Loss', 'Loss_Increase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for sigma in NOISE_LEVELS:
            writer.writerow({
                'Sigma': sigma,
                'Success_Rate': results[sigma]['successful_attacks'] / results[sigma]['total'],
                'Original_Accuracy': results[sigma]['original_accuracy'],
                'Noisy_Accuracy': results[sigma]['noisy_accuracy'],
                'Accuracy_Reduction': results[sigma]['original_accuracy'] - results[sigma]['noisy_accuracy'],
                'Original_Loss': results[sigma]['original_loss'],
                'Noisy_Loss': results[sigma]['noisy_loss'],
                'Loss_Increase': results[sigma]['noisy_loss'] - results[sigma]['original_loss']
            })
    
    print("\nResults saved to 'multi-label-classification/Adversarial/noise_results/metrics.csv'")
    print("Comparison plots saved to 'multi-label-classification/Adversarial/noise_results/metrics_comparison.png'")
    print("Example images saved to 'multi-label-classification/Adversarial/noise_results/examples/'")


# --- Main Script: Load model and test on a sample image ---
if __name__ == "__main__":
    main() 