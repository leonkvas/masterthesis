import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Parameters (must match training) ---
IMG_SIZE = (50, 250)  # (height, width)
CHANNELS = 1  # Grayscale images

# Model path
MODEL_PATH = "best_double_conv_layers_model.keras"

# Adversarial examples directory
ADV_EXAMPLES_DIR = "multi-label-classification/Adversarial/transferability_examples/fgsm/epsilon_3/test"

# Vocabulary settings
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# Determine Maximum CAPTCHA Length
file_list = [os.path.splitext(f)[0] for f in os.listdir(ADV_EXAMPLES_DIR) if f.endswith(('.png', '.jpg'))]
max_captcha_len = max(len(name) for name in file_list)

def load_and_preprocess_image(image_path):
    """Reads an image from 'image_path', resizes it to IMG_SIZE, and normalizes pixel values."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1)  # Ensure images are RGB
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    return img

def extract_label_from_filename(file_name):
    """Extract the original label from the filename."""
    label_str = os.path.splitext(file_name)[0]
    if "_fgsm_" in label_str:
        label_str = label_str.split("_fgsm_")[0]
    elif "_" in label_str:
        label_str = label_str.split("_")[0]
    return label_str

def load_model_with_custom_objects(model_path):
    """Load model with custom metrics properly registered."""
    try:
        custom_objects = {
            'full_sequence_accuracy': full_sequence_accuracy
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model.trainable = False
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise e

@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    """Custom metric to measure full sequence accuracy."""
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def test_fgsm_examples():
    # Get list of FGSM example files
    fgsm_files = [f for f in os.listdir(ADV_EXAMPLES_DIR) if f.endswith(('.png', '.jpg'))]
    print(f"Found {len(fgsm_files)} FGSM examples")
    print(f"Files: {fgsm_files}")
    
    # Load model
    model = load_model_with_custom_objects(MODEL_PATH)
    print("\nModel loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"Model name: {model.name}")
    print(f"Model config: {model.get_config()}")
    print("\nModel Weights:")
    for layer in model.layers:
        if layer.weights:
            print(f"\nLayer: {layer.name}")
            for weight in layer.weights:
                print(f"  {weight.name}: shape={weight.shape}, mean={tf.reduce_mean(weight).numpy():.6f}, std={tf.math.reduce_std(weight).numpy():.6f}")
    
    correct_predictions = 0
    total_chars = 0
    correct_chars = 0
    
    for file_name in fgsm_files:
        print(f"\n{'='*50}")
        print(f"Processing file: {file_name}")
        
        # Load adversarial image
        adv_path = os.path.join(ADV_EXAMPLES_DIR, file_name)
        adv_image = load_and_preprocess_image(adv_path)
        print(f"Image shape after preprocessing: {adv_image.shape}")
        print(f"Image value range: [{tf.reduce_min(adv_image)}, {tf.reduce_max(adv_image)}]")
        
        # Get original label from filename
        original_label = extract_label_from_filename(file_name)
        print(f"Extracted label: {original_label}")
        
        # Create padded sequence for comparison
        seq = [char_to_idx.get(char, 0) for char in original_label]
        seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
        print(f"Padded sequence: {seq_padded}")
        
        # Make prediction
        adv_img_batch = tf.expand_dims(adv_image, 0)
        print(f"Batch shape before prediction: {adv_img_batch.shape}")
        
        predictions = model.predict(adv_img_batch, verbose=0)
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions value range: [{np.min(predictions)}, {np.max(predictions)}]")
        print(f"Raw predictions mean: {np.mean(predictions)}")
        print(f"Raw predictions std: {np.std(predictions)}")
        
        pred_indices = tf.argmax(predictions, axis=-1).numpy()[0]
        print(f"Prediction indices: {pred_indices}")
        
        pred_decoded = [idx_to_char[idx] for idx in pred_indices if idx != 0]
        print(f"Decoded indices: {pred_decoded}")
        
        prediction = ''.join(pred_decoded)
        print(f"Final prediction: {prediction}")
        
        # Calculate metrics
        is_correct = prediction == original_label
        if is_correct:
            correct_predictions += 1
        
        # Character-level accuracy
        total_chars += len(original_label)
        correct_chars += sum(1 for a, b in zip(prediction, original_label) if a == b)
        
        # Print individual results
        print(f"Original: {original_label}, Predicted: {prediction}, Correct: {is_correct}")
        print(f"Character matches: {[a == b for a, b in zip(prediction, original_label)]}")
    
    # Calculate and print overall metrics
    total = len(fgsm_files)
    success_rate = correct_predictions / total if total > 0 else 0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"Full sequence accuracy: {success_rate:.2%} ({correct_predictions}/{total})")
    print(f"Character accuracy: {char_accuracy:.2%} ({correct_chars}/{total_chars})")

if __name__ == "__main__":
    test_fgsm_examples() 