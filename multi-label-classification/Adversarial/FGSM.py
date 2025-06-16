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
EPSILONS = [0.05, 0.1, 0.15]  # Increased perturbation magnitudes
CHANNELS = 1  # Grayscale images

# Vocabulary settings (same as during training)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
# If model was trained with a specific fixed max captcha length:
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


def create_adversarial_pattern(input_image, input_label, model, epsilon):
    """Create adversarial pattern using single-step FGSM."""
    input_image = tf.convert_to_tensor(input_image)
    input_image = tf.expand_dims(input_image, 0)  # Add batch dimension
    input_label = tf.expand_dims(input_label, 0)  # Add batch dimension

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        # Calculate loss for each position in the sequence
        loss = 0
        for i in range(prediction.shape[1]):
            pos_pred = prediction[:, i, :]
            pos_target = input_label[:, i]
            if pos_target[0] != 0:
                pos_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    pos_target, pos_pred, from_logits=False
                )
                loss += pos_loss
        non_padding_positions = tf.reduce_sum(tf.cast(input_label != 0, tf.float32))
        loss = loss / (non_padding_positions + 1e-8)
    # Get gradients
    gradient = tape.gradient(loss, input_image)
    # FGSM perturbation: single step
    signed_grad = tf.sign(gradient)
    adv_image = input_image + epsilon * signed_grad
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    perturbation = adv_image - input_image
    # Print statistics
    print(f"\nFinal Loss: {loss}")
    print(f"Perturbation Statistics:")
    print(f"Min: {tf.reduce_min(perturbation).numpy()}")
    print(f"Max: {tf.reduce_max(perturbation).numpy()}")
    print(f"Mean: {tf.reduce_mean(perturbation).numpy()}")
    print(f"Non-zero elements: {tf.math.count_nonzero(perturbation).numpy()}")
    return adv_image[0]  # Remove batch dimension


def apply_fgsm_attack(image, epsilon, model, label):
    """Apply single-step FGSM attack to create adversarial example."""
    adv_image = create_adversarial_pattern(image, label, model, epsilon)
    return adv_image


def predict_captcha(model, image_path):
    """
    Loads an image, applies FGSM to generate an adversarial example,
    and returns the prediction on the adversarial image.
    """
    # Load and preprocess the original image
    img = load_and_preprocess_image(image_path)
    # Extract the ground truth label from filename
    label_str = extract_label_from_filename(image_path)
    true_label = label_str  # For printing purposes
    true_seq = label_to_sequence(label_str)

    # Generate adversarial image
    adv_img = apply_fgsm_attack(img, EPSILONS[0], model, true_seq)

    # Perform prediction on the adversarial image
    adv_img_batch = tf.expand_dims(adv_img, axis=0)
    preds = model.predict(adv_img_batch, verbose=0)
    pred_indices = np.argmax(preds, axis=-1)[0]
    predicted_text = ''.join([idx_to_char.get(idx, '') for idx in pred_indices if idx != 0])

    return true_label, predicted_text, adv_img


def visualize_attack(original_image, adversarial_image, original_label, 
                    predicted_original, predicted_adv, epsilon, save_path=None):
    """Visualize original and adversarial images with predictions."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Original\nLabel: {original_label}\nPred: {predicted_original}')
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(1, 3, 2)
    plt.imshow(adversarial_image.numpy().squeeze(), cmap='gray')
    plt.title(f'Adversarial (ε={epsilon})\nPred: {predicted_adv}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(1, 3, 3)
    difference = np.abs(adversarial_image.numpy().squeeze() - original_image.numpy().squeeze())
    plt.imshow(difference, cmap='hot', vmin=0, vmax=1)  # Fixed color scale
    plt.title('Perturbation Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_model_robustness(model, test_dir, num_samples=5):
    """Test model robustness against FGSM attacks."""
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    test_files = test_files[:num_samples]  # Limit number of samples
    
    results = {epsilon: {'successful_attacks': 0, 'total': 0, 
                          'original_loss': 0, 'adversarial_loss': 0,
                          'original_accuracy': 0, 'adversarial_accuracy': 0} 
              for epsilon in EPSILONS}
    
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
        
        for epsilon in EPSILONS:
            # Create adversarial example
            adversarial_image = apply_fgsm_attack(original_image, epsilon, model, label_sequence)
            
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
                results[epsilon]['successful_attacks'] += 1
                
                # Save visualization for successful attacks
                save_path = f'multi-label-classification/Adversarial/fgsm_results/examples/epsilon_{epsilon}_{file}'
                visualize_attack(
                    original_image, adversarial_image, original_label,
                    predicted_original, predicted_adv, epsilon, save_path
                )
            
            # Update metrics
            results[epsilon]['total'] += 1
            results[epsilon]['original_loss'] += original_loss
            results[epsilon]['adversarial_loss'] += adv_loss
            results[epsilon]['original_accuracy'] += original_seq_accuracy
            results[epsilon]['adversarial_accuracy'] += adv_seq_accuracy
    
    # Calculate averages
    for epsilon in EPSILONS:
        if results[epsilon]['total'] > 0:
            results[epsilon]['original_loss'] /= results[epsilon]['total']
            results[epsilon]['adversarial_loss'] /= results[epsilon]['total']
            results[epsilon]['original_accuracy'] /= results[epsilon]['total']
            results[epsilon]['adversarial_accuracy'] /= results[epsilon]['total']
    
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
    os.makedirs('multi-label-classification/Adversarial/fgsm_results', exist_ok=True)
    # Create examples subdirectory for images
    os.makedirs('multi-label-classification/Adversarial/fgsm_results/examples', exist_ok=True)
    
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
    print("\nTesting model robustness against FGSM attacks...")
    results = test_model_robustness(model, test_dir)
    
    # Print results
    print("\nAttack Success Rates:")
    for epsilon in EPSILONS:
        success_rate = results[epsilon]['successful_attacks'] / results[epsilon]['total']
        print(f"ε = {epsilon}: {success_rate:.2%} ({results[epsilon]['successful_attacks']}/{results[epsilon]['total']})")
        
    # Print accuracy and loss metrics
    print("\nAccuracy and Loss Metrics:")
    for epsilon in EPSILONS:
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Original Accuracy: {results[epsilon]['original_accuracy']:.4f}")
        print(f"  Adversarial Accuracy: {results[epsilon]['adversarial_accuracy']:.4f}")
        print(f"  Accuracy Reduction: {results[epsilon]['original_accuracy'] - results[epsilon]['adversarial_accuracy']:.4f}")
        print(f"  Original Loss: {results[epsilon]['original_loss']:.4f}")
        print(f"  Adversarial Loss: {results[epsilon]['adversarial_loss']:.4f}")
        print(f"  Loss Increase: {results[epsilon]['adversarial_loss'] - results[epsilon]['original_loss']:.4f}")
    
    # Plot success rates
    plt.figure(figsize=(15, 5))
    
    # Plot attack success rates
    plt.subplot(1, 3, 1)
    epsilons = list(results.keys())
    success_rates = [results[eps]['successful_attacks'] / results[eps]['total'] for eps in epsilons]
    
    plt.plot(epsilons, success_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Attack Success Rate')
    plt.title('FGSM Attack Success Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot accuracy comparison
    plt.subplot(1, 3, 2)
    original_accuracies = [results[eps]['original_accuracy'] for eps in epsilons]
    adv_accuracies = [results[eps]['adversarial_accuracy'] for eps in epsilons]
    
    plt.plot(epsilons, original_accuracies, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(epsilons, adv_accuracies, 'o-', linewidth=2, markersize=8, label='Adversarial')
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot loss comparison
    plt.subplot(1, 3, 3)
    original_losses = [results[eps]['original_loss'] for eps in epsilons]
    adv_losses = [results[eps]['adversarial_loss'] for eps in epsilons]
    
    plt.plot(epsilons, original_losses, 'o-', linewidth=2, markersize=8, label='Original')
    plt.plot(epsilons, adv_losses, 'o-', linewidth=2, markersize=8, label='Adversarial')
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi-label-classification/Adversarial/fgsm_results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    import csv
    with open('multi-label-classification/Adversarial/fgsm_results/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epsilon', 'Success_Rate', 'Original_Accuracy', 'Adversarial_Accuracy', 
                     'Accuracy_Reduction', 'Original_Loss', 'Adversarial_Loss', 'Loss_Increase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for epsilon in EPSILONS:
            writer.writerow({
                'Epsilon': epsilon,
                'Success_Rate': results[epsilon]['successful_attacks'] / results[epsilon]['total'],
                'Original_Accuracy': results[epsilon]['original_accuracy'],
                'Adversarial_Accuracy': results[epsilon]['adversarial_accuracy'],
                'Accuracy_Reduction': results[epsilon]['original_accuracy'] - results[epsilon]['adversarial_accuracy'],
                'Original_Loss': results[epsilon]['original_loss'],
                'Adversarial_Loss': results[epsilon]['adversarial_loss'],
                'Loss_Increase': results[epsilon]['adversarial_loss'] - results[epsilon]['original_loss']
            })
    
    print("\nResults saved to 'multi-label-classification/Adversarial/fgsm_results/metrics.csv'")
    print("Comparison plots saved to 'multi-label-classification/Adversarial/fgsm_results/metrics_comparison.png'")
    print("Example images saved to 'multi-label-classification/Adversarial/fgsm_results/examples/'")


# --- Main Script: Load model and test on a sample image ---
if __name__ == "__main__":
    main()
