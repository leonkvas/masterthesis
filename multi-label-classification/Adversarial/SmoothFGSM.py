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
EPSILONS = [0.1, 0.5, 1, 2]  # Perturbation magnitudes
CHANNELS = 1  # Grayscale images

# Gaussian smoothing parameters
SMOOTHING_SIGMAS = [0.5, 1.0, 1.5, 2.0]  # Standard deviations for Gaussian blur
KERNEL_SIZE = 5  # Size of Gaussian kernel

# Vocabulary settings (same as during training)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
# If your model was trained with a specific fixed max captcha length:
MAX_CAPTCHA_LEN = 7  # adjust as per your training configuration


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


def apply_gaussian_smoothing(image, sigma):
    """
    Apply Gaussian blur to the image using manual convolution.
    
    Args:
        image: Input image tensor
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image tensor
    """
    # Convert image to tensor if it's not already
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # If image has no batch dimension, add it
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)  # Add channel dim
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)  # Add batch dim
    
    # Create Gaussian kernel
    kernel_size = KERNEL_SIZE
    x = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x, x)
    
    # Calculate 2D Gaussian kernel
    kernel = tf.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Add feature dimensions to kernel
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    
    # Apply convolution with Gaussian kernel
    smoothed_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Remove batch dimension if the original image didn't have it
    if len(tf.shape(image)) == 4 and image.shape[0] == 1:
        smoothed_image = smoothed_image[0]
    
    return smoothed_image


def create_adversarial_pattern(input_image, input_label, model):
    """Create adversarial pattern using PGD."""
    input_image = tf.convert_to_tensor(input_image)
    input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
    
    # Expand label to match model output shape
    input_label = tf.expand_dims(input_label, 0)  # Add batch dimension
    
    # Create a target label that's different from the input
    target_label = tf.roll(input_label, shift=1, axis=1)  # Shift all characters by one position
    
    # Initialize adversarial example
    adv_image = tf.identity(input_image)
    
    # PGD parameters
    num_steps = 10
    alpha = 0.01  # Step size
    
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            
            # Calculate loss for each position in the sequence
            loss = 0
            for i in range(prediction.shape[1]):
                pos_pred = prediction[:, i, :]
                pos_target = target_label[:, i]
                
                if pos_target[0] != 0:
                    pos_loss = tf.keras.losses.sparse_categorical_crossentropy(
                        pos_target, pos_pred, from_logits=False
                    )
                    loss += pos_loss
            
            non_padding_positions = tf.reduce_sum(tf.cast(target_label != 0, tf.float32))
            loss = loss / (non_padding_positions + 1e-8)
        
        # Get gradients
        gradient = tape.gradient(loss, adv_image)
        
        # Update adversarial example
        adv_image = adv_image + alpha * tf.sign(gradient)
        
        # Project back to valid range
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    
    # Calculate final perturbation
    perturbation = adv_image - input_image
    
    return perturbation


def apply_smooth_fgsm_attack(image, epsilon, sigma, model, label):
    """
    Apply Gaussian smoothing followed by FGSM attack.
    
    Args:
        image: Original image tensor
        epsilon: Perturbation magnitude for FGSM
        sigma: Standard deviation for Gaussian smoothing
        model: The model to attack
        label: Original label sequence
        
    Returns:
        Smoothed image and adversarial image
    """
    # Make a copy of the original image
    original_image = tf.identity(image)
    
    # First apply Gaussian smoothing
    smoothed_image = apply_gaussian_smoothing(image, sigma)
    
    # Make sure smoothed_image is a 3D tensor (height, width, channels)
    if len(smoothed_image.shape) == 4:  # If it has batch dimension
        smoothed_image = smoothed_image[0]
    
    # Then create adversarial pattern on the smoothed image
    perturbation = create_adversarial_pattern(smoothed_image, label, model)
    
    # Apply perturbation to the ORIGINAL image (not the smoothed one)
    # This creates a more effective attack that's resistant to smoothing defenses
    perturbation = tf.squeeze(perturbation)
    
    # Ensure the perturbation has the same shape as the original image
    if len(perturbation.shape) != len(original_image.shape):
        if len(original_image.shape) > len(perturbation.shape):
            # Add a channel dimension to perturbation if needed
            perturbation = tf.expand_dims(perturbation, -1)
    
    # Add the perturbation to the original image
    adv_image = original_image + epsilon * perturbation
    
    # Clip to maintain valid pixel range
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    
    return smoothed_image, adv_image


def visualize_attack(original_image, smoothed_image, adversarial_image, original_label, 
                    predicted_original, predicted_adv, epsilon, sigma, save_path=None):
    """Visualize original, smoothed, and adversarial images with predictions."""
    plt.figure(figsize=(20, 5))
    
    # Convert tensors to numpy arrays for visualization and ensure proper dimensions
    if isinstance(original_image, tf.Tensor):
        original_image = original_image.numpy()
    if isinstance(smoothed_image, tf.Tensor):
        smoothed_image = smoothed_image.numpy()
    if isinstance(adversarial_image, tf.Tensor):
        adversarial_image = adversarial_image.numpy()
    
    # Ensure all images are 2D (squeeze out batch and channel dimensions if needed)
    if len(original_image.shape) > 2:
        original_image = np.squeeze(original_image)
    if len(smoothed_image.shape) > 2:
        smoothed_image = np.squeeze(smoothed_image)
    if len(adversarial_image.shape) > 2:
        adversarial_image = np.squeeze(adversarial_image)
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original\nLabel: {original_label}\nPred: {predicted_original}')
    plt.axis('off')
    
    # Smoothed image
    plt.subplot(1, 4, 2)
    plt.imshow(smoothed_image, cmap='gray')
    plt.title(f'Smoothed (σ={sigma})')
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(1, 4, 3)
    plt.imshow(adversarial_image, cmap='gray')
    plt.title(f'Smoothed + FGSM (ε={epsilon})\nPred: {predicted_adv}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 4, 4)
    difference = np.abs(adversarial_image - original_image)
    plt.imshow(difference, cmap='hot', vmin=0, vmax=1)  # Fixed color scale
    plt.title('Perturbation Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_model_robustness(model, test_dir, num_samples=25):
    """Test model robustness against Smoothed FGSM attacks."""
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    test_files = test_files[:num_samples]  # Limit number of samples
    
    results = {}
    for sigma in SMOOTHING_SIGMAS:
        results[sigma] = {}
        for epsilon in EPSILONS:
            results[sigma][epsilon] = {
                'successful_attacks': 0, 
                'total': 0, 
                'original_loss': 0, 
                'adversarial_loss': 0,
                'original_accuracy': 0, 
                'adversarial_accuracy': 0
            }
    
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
        
        # Calculate original loss and accuracy
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
        
        for sigma in SMOOTHING_SIGMAS:
            for epsilon in EPSILONS:
                # Create adversarial example
                smoothed_image, adversarial_image = apply_smooth_fgsm_attack(
                    original_image, epsilon, sigma, model, label_sequence
                )
                
                # Get adversarial prediction
                adv_img_batch = tf.expand_dims(adversarial_image, axis=0)
                adv_preds = model.predict(adv_img_batch, verbose=0)
                adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
                predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
                
                # Calculate adversarial loss and accuracy
                adv_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    label_sequence_batch, adv_preds, from_logits=False
                )
                adv_loss = tf.reduce_mean(adv_loss).numpy()
                
                # Calculate character-level accuracy
                adv_correct = tf.cast(tf.equal(tf.cast(label_sequence_batch, tf.int64), 
                                             tf.argmax(adv_preds, axis=-1)), tf.float32)
                adv_char_accuracy = tf.reduce_mean(adv_correct).numpy()
                
                # Calculate sequence-level accuracy (full sequence correct)
                adv_seq_correct = tf.reduce_all(tf.equal(tf.cast(label_sequence_batch, tf.int64),
                                                      tf.argmax(adv_preds, axis=-1)), axis=1)
                adv_seq_accuracy = tf.reduce_mean(tf.cast(adv_seq_correct, tf.float32)).numpy()
                
                # Check if attack was successful
                if predicted_adv != true_label:
                    results[sigma][epsilon]['successful_attacks'] += 1
                    
                    # Save visualization for successful attacks
                    save_path = f'multi-label-classification/Adversarial/smooth_fgsm_results/example_images/smooth{sigma:.1f}_fgsm{epsilon:.1f}_{file}'
                    visualize_attack(
                        original_image, smoothed_image, adversarial_image, 
                        original_label, predicted_original, predicted_adv, 
                        epsilon, sigma, save_path
                    )
                
                # Update metrics
                results[sigma][epsilon]['total'] += 1
                results[sigma][epsilon]['original_loss'] += original_loss
                results[sigma][epsilon]['adversarial_loss'] += adv_loss
                results[sigma][epsilon]['original_accuracy'] += original_seq_accuracy
                results[sigma][epsilon]['adversarial_accuracy'] += adv_seq_accuracy
    
    # Calculate averages
    for sigma in SMOOTHING_SIGMAS:
        for epsilon in EPSILONS:
            if results[sigma][epsilon]['total'] > 0:
                results[sigma][epsilon]['original_loss'] /= results[sigma][epsilon]['total']
                results[sigma][epsilon]['adversarial_loss'] /= results[sigma][epsilon]['total']
                results[sigma][epsilon]['original_accuracy'] /= results[sigma][epsilon]['total']
                results[sigma][epsilon]['adversarial_accuracy'] /= results[sigma][epsilon]['total']
    
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
    os.makedirs('multi-label-classification/Adversarial/smooth_fgsm_results', exist_ok=True)
    # Create examples subdirectory
    os.makedirs('multi-label-classification/Adversarial/smooth_fgsm_results/example_images', exist_ok=True)
    
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
    print("\nTesting model robustness against Smoothed FGSM attacks...")
    results = test_model_robustness(model, test_dir)
    
    # Create CSV file for results
    import csv
    with open('multi-label-classification/Adversarial/smooth_fgsm_results/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Sigma', 'Epsilon', 'Success_Rate', 'Original_Accuracy', 'Adversarial_Accuracy', 
                     'Accuracy_Reduction', 'Original_Loss', 'Adversarial_Loss', 'Loss_Increase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Print results and write to CSV
        print("\nAttack Success Rates:")
        for sigma in SMOOTHING_SIGMAS:
            for epsilon in EPSILONS:
                success_rate = results[sigma][epsilon]['successful_attacks'] / results[sigma][epsilon]['total']
                acc_reduction = results[sigma][epsilon]['original_accuracy'] - results[sigma][epsilon]['adversarial_accuracy']
                loss_increase = results[sigma][epsilon]['adversarial_loss'] - results[sigma][epsilon]['original_loss']
                
                print(f"σ = {sigma}, ε = {epsilon}: {success_rate:.2%} ({results[sigma][epsilon]['successful_attacks']}/{results[sigma][epsilon]['total']})")
                print(f"  Original Accuracy: {results[sigma][epsilon]['original_accuracy']:.4f}")
                print(f"  Adversarial Accuracy: {results[sigma][epsilon]['adversarial_accuracy']:.4f}")
                print(f"  Accuracy Reduction: {acc_reduction:.4f}")
                print(f"  Original Loss: {results[sigma][epsilon]['original_loss']:.4f}")
                print(f"  Adversarial Loss: {results[sigma][epsilon]['adversarial_loss']:.4f}")
                print(f"  Loss Increase: {loss_increase:.4f}")
                
                writer.writerow({
                    'Sigma': sigma,
                    'Epsilon': epsilon,
                    'Success_Rate': success_rate,
                    'Original_Accuracy': results[sigma][epsilon]['original_accuracy'],
                    'Adversarial_Accuracy': results[sigma][epsilon]['adversarial_accuracy'],
                    'Accuracy_Reduction': acc_reduction,
                    'Original_Loss': results[sigma][epsilon]['original_loss'],
                    'Adversarial_Loss': results[sigma][epsilon]['adversarial_loss'],
                    'Loss_Increase': loss_increase
                })
    
    # Plot heatmap of success rates
    plt.figure(figsize=(10, 8))
    success_rates = np.zeros((len(SMOOTHING_SIGMAS), len(EPSILONS)))
    
    for i, sigma in enumerate(SMOOTHING_SIGMAS):
        for j, epsilon in enumerate(EPSILONS):
            success_rates[i, j] = results[sigma][epsilon]['successful_attacks'] / results[sigma][epsilon]['total']
    
    plt.imshow(success_rates, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attack Success Rate')
    plt.xticks(np.arange(len(EPSILONS)), [f'{eps}' for eps in EPSILONS])
    plt.yticks(np.arange(len(SMOOTHING_SIGMAS)), [f'{sig}' for sig in SMOOTHING_SIGMAS])
    plt.xlabel('Epsilon (FGSM Perturbation)')
    plt.ylabel('Sigma (Gaussian Smoothing)')
    plt.title('Smoothed FGSM Attack Success Rate')
    
    # Add text annotations
    for i in range(len(SMOOTHING_SIGMAS)):
        for j in range(len(EPSILONS)):
            plt.text(j, i, f'{success_rates[i, j]:.2f}', 
                     ha="center", va="center", color="w" if success_rates[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/smooth_fgsm_results/success_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracy reduction heatmap
    plt.figure(figsize=(10, 8))
    acc_reductions = np.zeros((len(SMOOTHING_SIGMAS), len(EPSILONS)))
    
    for i, sigma in enumerate(SMOOTHING_SIGMAS):
        for j, epsilon in enumerate(EPSILONS):
            acc_reductions[i, j] = results[sigma][epsilon]['original_accuracy'] - results[sigma][epsilon]['adversarial_accuracy']
    
    plt.imshow(acc_reductions, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Accuracy Reduction')
    plt.xticks(np.arange(len(EPSILONS)), [f'{eps}' for eps in EPSILONS])
    plt.yticks(np.arange(len(SMOOTHING_SIGMAS)), [f'{sig}' for sig in SMOOTHING_SIGMAS])
    plt.xlabel('Epsilon (FGSM Perturbation)')
    plt.ylabel('Sigma (Gaussian Smoothing)')
    plt.title('Accuracy Reduction in Smoothed FGSM Attack')
    
    # Add text annotations
    for i in range(len(SMOOTHING_SIGMAS)):
        for j in range(len(EPSILONS)):
            plt.text(j, i, f'{acc_reductions[i, j]:.2f}', 
                     ha="center", va="center", color="w" if acc_reductions[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/smooth_fgsm_results/accuracy_reduction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to 'multi-label-classification/Adversarial/smooth_fgsm_results/metrics.csv'")
    print("Heatmaps saved to 'multi-label-classification/Adversarial/smooth_fgsm_results/'")
    print("Example images saved to 'multi-label-classification/Adversarial/smooth_fgsm_results/example_images/'")


# --- Main Script: Load model and test with Smoothed FGSM attack ---
if __name__ == "__main__":
    main() 