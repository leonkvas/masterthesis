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
CHANNELS = 1  # Grayscale images

# Attack parameters
CONFIDENCE_LEVELS = [0.7, 0.8, 0.9, 0.95]  # Target confidence for the attack
MAX_ITERATIONS = 100  # Maximum number of iterations for the attack
STEP_SIZE = 0.01  # Step size for each iteration
EARLY_STOP = True  # Whether to stop early when target confidence is reached

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


def create_target_label(original_label):
    """
    Create a target label that's different from the original label.
    This implementation creates a target by replacing digits with similar-looking characters:
    0->O, 1->I, 2->Z, 5->S, 8->B, etc.
    """
    substitutions = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1',
        '2': 'Z', 'Z': '2',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '6': 'G', 'G': '6',
        '9': 'Q', 'Q': '9',
        'M': 'W', 'W': 'M'
    }
    
    target = ""
    for char in original_label:
        if char in substitutions:
            target += substitutions[char]
        else:
            # If no substitution, keep the original character
            target += char
    
    # Ensure target is different from original
    if target == original_label:
        if len(original_label) > 0:
            # Swap first and last character if no substitutions were made
            chars = list(original_label)
            chars[0], chars[-1] = chars[-1], chars[0]
            target = ''.join(chars)
    
    return target


def apply_ifgs_attack(image, original_label, target_label, model, confidence_level):
    """
    Apply Iterative Fast Gradient Sign (IFGS) attack to create an adversarial example
    that is classified as the target label with the desired confidence level.
    
    Args:
        image: Original image tensor
        original_label: Original label string
        target_label: Target label string
        model: Trained model
        confidence_level: Desired confidence level for the target prediction
        
    Returns:
        adversarial_image: The adversarial example
        iteration: Number of iterations taken
        confidence: Final confidence level achieved
    """
    # Convert labels to sequences
    original_seq = label_to_sequence(original_label)
    target_seq = label_to_sequence(target_label)
    
    # Add batch dimension
    image = tf.expand_dims(tf.convert_to_tensor(image), 0)
    target_seq = tf.expand_dims(target_seq, 0)
    
    # Initialize adversarial example
    adv_image = tf.identity(image)
    
    # Cumulative perturbation tracking
    total_perturbation = 0.0
    
    # Iteration counter
    iteration = 0
    
    # Track current confidence
    current_confidence = 0.0
    
    while iteration < MAX_ITERATIONS and current_confidence < confidence_level:
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            
            # Calculate loss for each position in the sequence
            loss = 0
            confidence_sum = 0
            non_padding_positions = 0
            
            for i in range(prediction.shape[1]):
                pos_pred = prediction[:, i, :]
                pos_target = target_seq[:, i]
                
                if pos_target[0] != 0:
                    # Calculate loss to maximize confidence in target label
                    pos_target_confidence = pos_pred[0, pos_target[0]]
                    confidence_sum += pos_target_confidence
                    non_padding_positions += 1
                    
                    # Negative cross-entropy loss to maximize probability of target
                    pos_loss = -tf.math.log(pos_target_confidence + 1e-10)
                    loss += pos_loss
            
            # Average confidence across non-padding positions
            current_confidence = confidence_sum / non_padding_positions if non_padding_positions > 0 else 0
            
            # Average loss
            loss = loss / max(non_padding_positions, 1)
        
        # Calculate gradients
        gradients = tape.gradient(loss, adv_image)
        
        # Update adversarial example using gradient sign
        perturbation = STEP_SIZE * tf.sign(gradients)
        adv_image = adv_image - perturbation  # Subtract because we're maximizing confidence
        
        # Clip to maintain valid pixel range [0, 1]
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)
        
        # Track total perturbation magnitude
        total_perturbation += tf.reduce_mean(tf.abs(perturbation))
        
        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Confidence = {current_confidence:.4f}, Loss = {loss:.4f}")
        
        # Early stopping if target confidence is reached
        if EARLY_STOP and current_confidence >= confidence_level:
            print(f"Target confidence reached at iteration {iteration}")
            break
        
        iteration += 1
    
    # Calculate final perturbation
    perturbation = adv_image - image
    print(f"\nAttack completed in {iteration} iterations")
    print(f"Final confidence: {current_confidence:.4f}")
    print(f"Total perturbation magnitude: {total_perturbation:.4f}")
    print(f"Perturbation statistics:")
    print(f"  Min: {tf.reduce_min(perturbation).numpy()}")
    print(f"  Max: {tf.reduce_max(perturbation).numpy()}")
    print(f"  Mean: {tf.reduce_mean(tf.abs(perturbation)).numpy()}")
    
    return adv_image[0], iteration, current_confidence.numpy()


def visualize_attack(original_image, adversarial_image, original_label, target_label, 
                     predicted_adv, confidence, iterations, save_path=None):
    """Visualize original and adversarial images with predictions."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {original_label}')
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(1, 3, 2)
    plt.imshow(adversarial_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Adversarial\nTarget: {target_label}\nPred: {predicted_adv}\nConf: {confidence:.2f}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 3, 3)
    difference = np.abs(adversarial_image.numpy().squeeze() - original_image.numpy().squeeze())
    plt.imshow(difference, cmap='hot', vmin=0, vmax=1)  # Fixed color scale
    plt.title(f'Perturbation Map\nIterations: {iterations}')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_model_robustness(model, test_dir, num_samples=50):
    """Test model robustness against IFGS attacks."""
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    test_files = test_files[:num_samples]  # Limit number of samples
    
    results = {confidence: {'successful_attacks': 0, 'total': 0, 'avg_iterations': 0,
                            'original_loss': 0, 'adversarial_loss': 0,
                            'original_accuracy': 0, 'adversarial_accuracy': 0} 
              for confidence in CONFIDENCE_LEVELS}
    
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
        
        print(f"\nProcessing {file}: Original label = {original_label}, Prediction = {predicted_original}")
        
        for confidence in CONFIDENCE_LEVELS:
            print(f"Applying IFGS attack with confidence = {confidence}...")
            
            # Apply IFGS attack
            adv_image, iterations, achieved_confidence = apply_ifgs_attack(original_image, original_label, create_target_label(original_label), model, confidence)
            
            # Get prediction on adversarial image
            adv_img_batch = tf.expand_dims(adv_image, axis=0)
            adv_preds = model.predict(adv_img_batch, verbose=0)
            adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
            predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
            
            # Calculate adversarial loss
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
            
            print(f"  Original prediction: {predicted_original}")
            print(f"  Adversarial prediction: {predicted_adv} (after {iterations} iterations)")
            
            # Check if attack was successful
            if predicted_adv != true_label:
                results[confidence]['successful_attacks'] += 1
                results[confidence]['avg_iterations'] += iterations
                
                # Save visualization for successful attacks
                save_path = f'multi-label-classification/Adversarial/ifgs_results/examples/conf_{confidence}_{file}'
                visualize_attack(
                    original_image, adv_image, original_label,
                    create_target_label(original_label), predicted_adv, confidence, iterations, save_path
                )
            
            # Update metrics
            results[confidence]['total'] += 1
            results[confidence]['original_loss'] += original_loss
            results[confidence]['adversarial_loss'] += adv_loss
            results[confidence]['original_accuracy'] += original_seq_accuracy
            results[confidence]['adversarial_accuracy'] += adv_seq_accuracy
    
    # Calculate averages
    for confidence in CONFIDENCE_LEVELS:
        if results[confidence]['total'] > 0:
            if results[confidence]['successful_attacks'] > 0:
                results[confidence]['avg_iterations'] /= results[confidence]['successful_attacks']
            results[confidence]['original_loss'] /= results[confidence]['total']
            results[confidence]['adversarial_loss'] /= results[confidence]['total']
            results[confidence]['original_accuracy'] /= results[confidence]['total']
            results[confidence]['adversarial_accuracy'] /= results[confidence]['total']
    
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
    os.makedirs('multi-label-classification/Adversarial/ifgs_results', exist_ok=True)
    # Create examples subdirectory
    os.makedirs('multi-label-classification/Adversarial/ifgs_results/examples', exist_ok=True)
    
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
    print("\nTesting model robustness against IFGS attacks...")
    results = test_model_robustness(model, test_dir)
    
    # Print results
    print("\nAttack Success Rates:")
    for confidence in CONFIDENCE_LEVELS:
        success_rate = results[confidence]['successful_attacks'] / results[confidence]['total']
        avg_iterations = results[confidence]['avg_iterations'] if results[confidence]['successful_attacks'] > 0 else 0
        print(f"Confidence = {confidence}: {success_rate:.2%} ({results[confidence]['successful_attacks']}/{results[confidence]['total']})")
        if results[confidence]['successful_attacks'] > 0:
            print(f"  Average iterations for successful attacks: {avg_iterations:.1f}")
    
    # Print accuracy and loss metrics
    print("\nAccuracy and Loss Metrics:")
    for confidence in CONFIDENCE_LEVELS:
        print(f"\nConfidence = {confidence}:")
        print(f"  Original Accuracy: {results[confidence]['original_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results[confidence]['adversarial_accuracy']:.4f}")
        print(f"  Accuracy Reduction: {results[confidence]['original_accuracy'] - results[confidence]['adversarial_accuracy']:.4f}")
        print(f"  Original Loss: {results[confidence]['original_loss']:.4f}")
        print(f"  Adversarial Loss: {results[confidence]['adversarial_loss']:.4f}")
        print(f"  Loss Increase: {results[confidence]['adversarial_loss'] - results[confidence]['original_loss']:.4f}")
    
    # Plot success rates
    plt.figure(figsize=(15, 8))
    
    # Plot attack success rates
    plt.subplot(2, 2, 1)
    confidences = list(results.keys())
    success_rates = [results[conf]['successful_attacks'] / results[conf]['total'] for conf in confidences]
    
    plt.plot(confidences, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Target Confidence Level')
    plt.ylabel('Attack Success Rate')
    plt.title('IFGS Success Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot average iterations
    plt.subplot(2, 2, 2)
    avg_iterations = [results[conf]['avg_iterations'] if results[conf]['successful_attacks'] > 0 else 0 for conf in confidences]
    
    plt.plot(confidences, avg_iterations, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Target Confidence Level')
    plt.ylabel('Average Iterations')
    plt.title('IFGS Average Iterations')
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 3)
    original_accuracies = [results[conf]['original_accuracy'] for conf in confidences]
    adv_accuracies = [results[conf]['adversarial_accuracy'] for conf in confidences]
    
    plt.plot(confidences, original_accuracies, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(confidences, adv_accuracies, 'o-', linewidth=2, markersize=8, label='Adversarial')
    plt.xlabel('Target Confidence Level')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot loss comparison
    plt.subplot(2, 2, 4)
    original_losses = [results[conf]['original_loss'] for conf in confidences]
    adv_losses = [results[conf]['adversarial_loss'] for conf in confidences]
    
    plt.plot(confidences, original_losses, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(confidences, adv_losses, 'o-', linewidth=2, markersize=8, label='Adversarial')
    plt.xlabel('Target Confidence Level')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/ifgs_results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    import csv
    with open('multi-label-classification/Adversarial/ifgs_results/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Confidence', 'Success_Rate', 'Avg_Iterations', 'Original_Accuracy', 'Adversarial_Accuracy', 
                    'Accuracy_Reduction', 'Original_Loss', 'Adversarial_Loss', 'Loss_Increase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for confidence in CONFIDENCE_LEVELS:
            writer.writerow({
                'Confidence': confidence,
                'Success_Rate': results[confidence]['successful_attacks'] / results[confidence]['total'],
                'Avg_Iterations': results[confidence]['avg_iterations'],
                'Original_Accuracy': results[confidence]['original_accuracy'],
                'Adversarial_Accuracy': results[confidence]['adversarial_accuracy'],
                'Accuracy_Reduction': results[confidence]['original_accuracy'] - results[confidence]['adversarial_accuracy'],
                'Original_Loss': results[confidence]['original_loss'],
                'Adversarial_Loss': results[confidence]['adversarial_loss'],
                'Loss_Increase': results[confidence]['adversarial_loss'] - results[confidence]['original_loss']
            })
    
    print("\nResults saved to 'multi-label-classification/Adversarial/ifgs_results/metrics.csv'")
    print("Comparison plots saved to 'multi-label-classification/Adversarial/ifgs_results/metrics_comparison.png'")


# --- Main Script: Load model and test with IFGS/IAN attack ---
if __name__ == "__main__":
    main() 