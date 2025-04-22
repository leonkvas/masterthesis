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
EPSILON = 0.1  # Maximum perturbation magnitude
ALPHA = 0.01  # Step size for each iteration
NUM_ITERATIONS = 40  # Number of iterations for the attack
RANDOM_START = True  # Whether to add random noise at the beginning

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


def apply_bim_aecaptcha(image, original_label, model):
    """
    Apply Basic Iterative Method (BIM) to create an adversarial example
    that maximizes the loss for the entire CAPTCHA sequence.
    
    This implementation targets all characters at once, creating multi-character
    disturbances in a single optimization process.
    
    Args:
        image: Original image tensor
        original_label: Original label string
        model: Trained model
        
    Returns:
        adversarial_image: The adversarial example
        iteration: Number of iterations taken
        loss_change: Change in loss from original to adversarial
    """
    # Convert label to sequence
    original_seq = label_to_sequence(original_label)
    
    # Add batch dimension
    image = tf.expand_dims(tf.convert_to_tensor(image), 0)
    original_seq = tf.expand_dims(original_seq, 0)
    
    # Get original loss
    prediction = model(image)
    original_loss = tf.keras.losses.sparse_categorical_crossentropy(
        original_seq, prediction, from_logits=False
    )
    original_loss = tf.reduce_mean(original_loss)
    
    # Initialize adversarial example
    if RANDOM_START:
        # Add small random noise to start
        random_noise = tf.random.uniform(image.shape, -EPSILON/2, EPSILON/2)
        adv_image = tf.clip_by_value(image + random_noise, 0.0, 1.0)
    else:
        adv_image = tf.identity(image)
    
    # Iteration counter
    iteration = 0
    current_loss = original_loss
    
    # Used to track progress
    losses = []
    
    # Store the best adversarial example (with highest loss)
    best_adv_image = tf.identity(adv_image)
    best_loss = current_loss
    
    for i in range(NUM_ITERATIONS):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            
            # Calculate loss for all character positions at once
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                original_seq, prediction, from_logits=False
            )
            # Average across sequence positions
            loss = tf.reduce_mean(loss)
        
        # Get gradients
        gradients = tape.gradient(loss, adv_image)
        
        # Update adversarial example using gradient sign (maximize loss)
        perturbation = ALPHA * tf.sign(gradients)
        adv_image = adv_image + perturbation
        
        # Clip the perturbation to maintain epsilon constraint
        delta = tf.clip_by_value(adv_image - image, -EPSILON, EPSILON)
        adv_image = image + delta
        
        # Clip to maintain valid pixel range [0, 1]
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)
        
        # Calculate current loss
        prediction = model(adv_image)
        current_loss = tf.keras.losses.sparse_categorical_crossentropy(
            original_seq, prediction, from_logits=False
        )
        current_loss = tf.reduce_mean(current_loss)
        losses.append(current_loss.numpy())
        
        # Track the best adversarial example
        if current_loss > best_loss:
            best_adv_image = tf.identity(adv_image)
            best_loss = current_loss
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{NUM_ITERATIONS}: Loss = {current_loss:.4f} (Original: {original_loss:.4f})")
        
        iteration += 1
    
    # Use the best adversarial example
    adv_image = best_adv_image
    
    # Calculate the total change in loss
    loss_change = best_loss - original_loss
    
    # Calculate final perturbation
    perturbation = adv_image - image
    print(f"\nAttack completed in {iteration} iterations")
    print(f"Original loss: {original_loss:.4f}")
    print(f"Final loss: {best_loss:.4f}")
    print(f"Loss change: {loss_change:.4f} ({loss_change/original_loss:.2%} increase)")
    print(f"Perturbation statistics:")
    print(f"  Min: {tf.reduce_min(perturbation).numpy()}")
    print(f"  Max: {tf.reduce_max(perturbation).numpy()}")
    print(f"  Mean: {tf.reduce_mean(tf.abs(perturbation)).numpy()}")
    
    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss during BIM attack')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/bim_results/loss_curve.png')
    plt.close()
    
    return adv_image[0], iteration, loss_change.numpy()


def visualize_attack(original_image, adversarial_image, original_label, 
                    original_pred, adversarial_pred, loss_change, save_path=None):
    """Visualize original and adversarial images with predictions."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {original_label}\nPred: {original_pred}')
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(1, 3, 2)
    plt.imshow(adversarial_image.numpy().squeeze(), cmap='gray')
    plt.title(f'AECAPTCHA\nPred: {adversarial_pred}\nLoss change: {loss_change:.2f}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 3, 3)
    difference = np.abs(adversarial_image.numpy().squeeze() - original_image.numpy().squeeze())
    plt.imshow(difference, cmap='hot', vmin=0, vmax=EPSILON)  # Fixed color scale
    plt.title('Perturbation Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def character_accuracy(true_label, predicted_label):
    """Calculate character accuracy between two strings."""
    min_len = min(len(true_label), len(predicted_label))
    correct = sum(true_label[i] == predicted_label[i] for i in range(min_len))
    return correct / len(true_label) if len(true_label) > 0 else 0


def test_model_robustness(model, test_dir, num_samples=50):
    """Test model robustness against AECAPTCHA using BIM attack."""
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    test_files = test_files[:num_samples]  # Limit number of samples
    
    overall_results = {
        'total_samples': 0,
        'successful_attacks': 0,
        'total_char_error_increase': 0,
        'avg_loss_change': 0,
        'avg_iterations': 0,
        'avg_original_loss': 0,
        'avg_adversarial_loss': 0,
        'avg_original_accuracy': 0,
        'avg_adversarial_accuracy': 0
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
        original_pred = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
        
        # Check if original prediction is correct, skip if not
        if original_pred != original_label:
            print(f"Skipping {file}: original prediction '{original_pred}' doesn't match label '{original_label}'")
            skipped_samples += 1
            continue
        
        processed_samples += 1
        
        # Calculate original loss
        original_loss = tf.keras.losses.sparse_categorical_crossentropy(
            label_sequence_batch, original_preds, from_logits=False
        )
        original_loss = tf.reduce_mean(original_loss).numpy()
        
        # Calculate original character accuracy
        original_correct = tf.cast(tf.equal(tf.cast(label_sequence_batch, tf.int64), 
                                         tf.argmax(original_preds, axis=-1)), tf.float32)
        original_char_acc = tf.reduce_mean(original_correct).numpy()
        
        # Calculate original sequence accuracy
        original_seq_correct = tf.reduce_all(tf.equal(tf.cast(label_sequence_batch, tf.int64),
                                                   tf.argmax(original_preds, axis=-1)), axis=1)
        original_seq_acc = tf.reduce_mean(tf.cast(original_seq_correct, tf.float32)).numpy()
        
        print(f"\nProcessing {file}: Original label = {original_label}, Prediction = {original_pred}")
        
        # Apply attack
        adv_image, iterations, loss_change = apply_bim_aecaptcha(
            original_image, original_label, model
        )
        
        # Get prediction on adversarial image
        adv_img_batch = tf.expand_dims(adv_image, 0)
        adv_preds = model.predict(adv_img_batch, verbose=0)
        adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
        adversarial_pred = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
        
        # Calculate adversarial loss
        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(
            label_sequence_batch, adv_preds, from_logits=False
        )
        adv_loss = tf.reduce_mean(adv_loss).numpy()
        
        # Calculate adversarial character accuracy
        adv_correct = tf.cast(tf.equal(tf.cast(label_sequence_batch, tf.int64), 
                                    tf.argmax(adv_preds, axis=-1)), tf.float32)
        adv_char_acc = tf.reduce_mean(adv_correct).numpy()
        
        # Calculate adversarial sequence accuracy
        adv_seq_correct = tf.reduce_all(tf.equal(tf.cast(label_sequence_batch, tf.int64),
                                              tf.argmax(adv_preds, axis=-1)), axis=1)
        adv_seq_acc = tf.reduce_mean(tf.cast(adv_seq_correct, tf.float32)).numpy()
        
        # Calculate character error increase
        char_error_increase = original_char_acc - adv_char_acc
        
        print(f"Adversarial Prediction: {adversarial_pred}")
        print(f"Character Error Increase: {char_error_increase:.2%}")
        print(f"Original Loss: {original_loss:.4f}, Adversarial Loss: {adv_loss:.4f}")
        print(f"Original Sequence Accuracy: {original_seq_acc:.4f}, Adversarial Sequence Accuracy: {adv_seq_acc:.4f}")
        
        # Check if attack was successful (prediction changed)
        success = (adversarial_pred != original_pred)
        
        if success:
            overall_results['successful_attacks'] += 1
            
            # Save visualization for successful attacks
            save_path = f'multi-label-classification/Adversarial/bim_results/example_images/aecaptcha_{file}'
            visualize_attack(
                original_image, adv_image, original_label,
                original_pred, adversarial_pred, loss_change, save_path
            )
        
        overall_results['total_samples'] += 1
        overall_results['total_char_error_increase'] += char_error_increase
        overall_results['avg_loss_change'] += loss_change
        overall_results['avg_iterations'] += iterations
        overall_results['avg_original_loss'] += original_loss
        overall_results['avg_adversarial_loss'] += adv_loss
        overall_results['avg_original_accuracy'] += original_seq_acc
        overall_results['avg_adversarial_accuracy'] += adv_seq_acc
    
    # Calculate averages
    if overall_results['total_samples'] > 0:
        overall_results['avg_char_error_increase'] = overall_results['total_char_error_increase'] / overall_results['total_samples']
        overall_results['avg_loss_change'] = overall_results['avg_loss_change'] / overall_results['total_samples']
        overall_results['avg_iterations'] = overall_results['avg_iterations'] / overall_results['total_samples']
        overall_results['avg_original_loss'] /= overall_results['total_samples']
        overall_results['avg_adversarial_loss'] /= overall_results['total_samples']
        overall_results['avg_original_accuracy'] /= overall_results['total_samples']
        overall_results['avg_adversarial_accuracy'] /= overall_results['total_samples']
    
    # Print summary of processed vs skipped samples
    print(f"\nSummary of processed samples:")
    print(f"  Total images checked: {len(test_files)}")
    print(f"  Images with correct original predictions: {processed_samples}")
    print(f"  Images skipped (incorrect original predictions): {skipped_samples}")
    
    return overall_results


@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def main():
    # Create results directory
    os.makedirs('multi-label-classification/Adversarial/bim_results', exist_ok=True)
    # Create examples subdirectory
    os.makedirs('multi-label-classification/Adversarial/bim_results/example_images', exist_ok=True)
    
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
    print("\nTesting model robustness against AECAPTCHA using BIM...")
    results = test_model_robustness(model, test_dir)
    
    # Print overall results
    print("\nOverall Results:")
    print(f"Success Rate: {results['successful_attacks'] / results['total_samples']:.2%} ({results['successful_attacks']}/{results['total_samples']})")
    print(f"Average Character Error Increase: {results['avg_char_error_increase']:.2%}")
    print(f"Average Loss Change: {results['avg_loss_change']:.4f}")
    print(f"Average Iterations: {results['avg_iterations']:.1f}")
    
    # Print accuracy and loss metrics
    print("\nAccuracy and Loss Metrics:")
    print(f"  Original Accuracy: {results['avg_original_accuracy']:.4f}")
    print(f"  Adversarial Accuracy: {results['avg_adversarial_accuracy']:.4f}")
    print(f"  Accuracy Reduction: {results['avg_original_accuracy'] - results['avg_adversarial_accuracy']:.4f}")
    print(f"  Original Loss: {results['avg_original_loss']:.4f}")
    print(f"  Adversarial Loss: {results['avg_adversarial_loss']:.4f}")
    print(f"  Loss Increase: {results['avg_adversarial_loss'] - results['avg_original_loss']:.4f}")
    
    # Create a comprehensive summary chart
    plt.figure(figsize=(15, 10))
    
    # Success rate and error increase
    plt.subplot(2, 2, 1)
    metrics = ['Success Rate', 'Error Increase', 'Acc Reduction']
    values = [
        results['successful_attacks'] / results['total_samples'],
        results['avg_char_error_increase'],
        results['avg_original_accuracy'] - results['avg_adversarial_accuracy']
    ]
    
    plt.bar(metrics, values, color=['red', 'orange', 'blue'])
    plt.ylim(0, 1)
    plt.title('Attack Effectiveness')
    plt.grid(True, alpha=0.3)
    
    # Loss comparison
    plt.subplot(2, 2, 2)
    loss_labels = ['Original Loss', 'Adversarial Loss']
    loss_values = [results['avg_original_loss'], results['avg_adversarial_loss']]
    
    plt.bar(loss_labels, loss_values, color=['green', 'purple'])
    plt.title('Loss Comparison')
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison
    plt.subplot(2, 2, 3)
    acc_labels = ['Original Accuracy', 'Adversarial Accuracy']
    acc_values = [results['avg_original_accuracy'], results['avg_adversarial_accuracy']]
    
    plt.bar(acc_labels, acc_values, color=['green', 'purple'])
    plt.ylim(0, 1)
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    
    # Loss change distribution (if we had multiple epsilon values)
    plt.subplot(2, 2, 4)
    plt.hist([results['avg_loss_change']], bins=10, alpha=0.7, color='blue')
    plt.axvline(results['avg_loss_change'], color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Loss Change')
    plt.ylabel('Frequency')
    plt.title('Loss Change Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/bim_results/comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    import csv
    with open('multi-label-classification/Adversarial/bim_results/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({'Metric': 'Success_Rate', 'Value': results['successful_attacks'] / results['total_samples']})
        writer.writerow({'Metric': 'Avg_Char_Error_Increase', 'Value': results['avg_char_error_increase']})
        writer.writerow({'Metric': 'Avg_Loss_Change', 'Value': results['avg_loss_change']})
        writer.writerow({'Metric': 'Avg_Iterations', 'Value': results['avg_iterations']})
        writer.writerow({'Metric': 'Original_Accuracy', 'Value': results['avg_original_accuracy']})
        writer.writerow({'Metric': 'Adversarial_Accuracy', 'Value': results['avg_adversarial_accuracy']})
        writer.writerow({'Metric': 'Accuracy_Reduction', 'Value': results['avg_original_accuracy'] - results['avg_adversarial_accuracy']})
        writer.writerow({'Metric': 'Original_Loss', 'Value': results['avg_original_loss']})
        writer.writerow({'Metric': 'Adversarial_Loss', 'Value': results['avg_adversarial_loss']})
        writer.writerow({'Metric': 'Loss_Increase', 'Value': results['avg_adversarial_loss'] - results['avg_original_loss']})
    
    print("\nResults saved to 'multi-label-classification/Adversarial/bim_results/metrics.csv'")
    print("Comprehensive metrics chart saved to 'multi-label-classification/Adversarial/bim_results/comprehensive_metrics.png'")


# --- Main Script: Load model and test with AECAPTCHA using BIM ---
if __name__ == "__main__":
    main() 