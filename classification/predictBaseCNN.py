import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
IMG_SIZE = (50, 250)  # As defined in your original training setup
BATCH_SIZE = 16

# Test directory 
test_data_dir = "data/test2_organized/light_blue"
# test on adversarial examples
#test_data_dir = "data/train_2_adversarial_examples/bim/epsilon_0.1"
#test_data_dir = "multi-label-classification/Adversarial/bim_results/example_images"
#test_data_dir = "multi-label-classification/Adversarial/transferability_examples/fgsm/epsilon_3"

# Define the vocabulary and mapping
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}  # Mapping: '0'->1, etc.
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# Determine Maximum CAPTCHA Length
file_list = [os.path.splitext(f)[0] for f in os.listdir(test_data_dir) if f.endswith(('.png', '.jpg'))]
max_captcha_len = max(len(name) for name in file_list)


# --- Helper function to preprocess test images ---
def load_test_image_and_label(file_path):
    """
    Loads an image and its corresponding label for testing.
    """
    # Convert EagerTensor to string
    file_path = file_path.numpy().decode('utf-8')

    # Load and preprocess image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1)  # Ensure images are RGB
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize

    # Extract label from the file name
    file_name = os.path.basename(file_path)
    label_str = os.path.splitext(file_name)[0]
    if "_fgsm_" in label_str:
        label_str = label_str.split("_fgsm_")[0]
    elif "_" in label_str:
        label_str = label_str.split("_")[0]
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
    return img, seq_padded


def create_test_dataset(test_data_dir):
    """
    Creates a test dataset from the test directory.
    """
    # Get list of test image file paths
    test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith(('.png', '.jpg'))]

    # Create a dataset
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)

    def process_file(file_path):
        img, label = tf.py_function(
            func=load_test_image_and_label,  # Your updated function
            inp=[file_path],  # Input argument to your function
            Tout=(tf.float32, tf.int32)  # Output types for the function
        )
        img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))  # Explicitly set tensor shape
        label.set_shape((max_captcha_len,))  # Shape of the padded label sequence
        return img, label

    # Apply the function and batch the dataset
    test_ds = test_ds.map(process_file, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return test_ds


def calculate_test_accuracy(model_path, test_data_dir):
    """
    Loads a pre-trained model and calculates prediction accuracy on the test dataset.
    Tracks and visualizes misclassified characters and their replacements.
    """
    # Load the model with custom metrics
    custom_objects = {
        'full_sequence_accuracy': full_sequence_accuracy
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    model.trainable = False
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
    
    # Get list of test files
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(('.png', '.jpg'))]
    print(f"\nFound {len(test_files)} test examples")
    print(f"Files: {test_files}")

    # Initialize counters for error analysis and metrics
    correct_predictions = 0
    total_predictions = 0
    correct_chars = 0
    total_chars = 0
    error_counter = Counter()  # Track misclassifications as ('True', 'Predicted'): count
    label_counter = Counter()  # Track how often each true character appears
    prediction_times = []  # Track time taken for each prediction

    for file_name in test_files:
        print(f"\n{'='*50}")
        print(f"Processing file: {file_name}")
        
        # Load and preprocess image
        file_path = os.path.join(test_data_dir, file_name)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=1)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        print(f"Image shape after preprocessing: {img.shape}")
        print(f"Image value range: [{tf.reduce_min(img)}, {tf.reduce_max(img)}]")

        # Extract label from filename
        label_str = os.path.splitext(file_name)[0]
        if "_fgsm_" in label_str:
            label_str = label_str.split("_fgsm_")[0]
        elif "_" in label_str:
            label_str = label_str.split("_")[0]
        print(f"Extracted label: {label_str}")
        
        seq = [char_to_idx.get(char, 0) for char in label_str]
        seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
        print(f"Padded sequence: {seq_padded}")

        # Make prediction
        img_batch = tf.expand_dims(img, 0)
        print(f"Batch shape before prediction: {img_batch.shape}")
        
        # Time the prediction
        start_time = tf.timestamp()
        predictions = model.predict(img_batch, verbose=0)
        end_time = tf.timestamp()
        prediction_time = (end_time - start_time).numpy() * 1000  # Convert to milliseconds
        prediction_times.append(prediction_time)
        
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions value range: [{np.min(predictions)}, {np.max(predictions)}]")
        print(f"Raw predictions mean: {np.mean(predictions)}")
        print(f"Raw predictions std: {np.std(predictions)}")
        
        pred_indices = tf.argmax(predictions, axis=-1).numpy()[0]
        print(f"Prediction indices: {pred_indices}")

        # Remove padding and decode
        pred_decoded = [idx_to_char[idx] for idx in pred_indices if idx != 0]
        print(f"Decoded indices: {pred_decoded}")
        
        true_decoded = [idx_to_char[idx] for idx in seq_padded if idx != 0]
        prediction = ''.join(pred_decoded)
        print(f"Final prediction: {prediction}")

        # Compare full label sequence
        if ''.join(pred_decoded) == ''.join(true_decoded):
            correct_predictions += 1
        else:
            print(f"Predicted: {''.join(pred_decoded)}, True: {''.join(true_decoded)}")
            print(f"Character matches: {[a == b for a, b in zip(prediction, ''.join(true_decoded))]}")

        # Track per-character accuracy
        total_chars += len(true_decoded)
        correct_chars += sum(1 for a, b in zip(pred_decoded, true_decoded) if a == b)

        # Track per-character misclassifications
        for true_char, pred_char in zip(true_decoded, pred_decoded):
            label_counter[true_char] += 1
            if true_char != pred_char:
                error_counter[(true_char, pred_char)] += 1

        total_predictions += 1

    # Calculate accuracies
    sequence_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    character_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    avg_prediction_time = np.mean(prediction_times) if prediction_times else 0

    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"Total examples tested: {total_predictions}")
    print(f"Full sequence accuracy: {sequence_accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")
    print(f"Character-level accuracy: {character_accuracy * 100:.2f}% ({correct_chars}/{total_chars})")
    print(f"Average prediction time: {avg_prediction_time:.2f} ms per image")
    print(f"{'='*50}")

    # Most frequent misclassifications
    print("\nMost Frequent Misclassifications:")
    for (true_char, pred_char), count in error_counter.most_common(10):
        print(f"  '{true_char}' misclassified as '{pred_char}': {count} times.")


@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# --- Run the Accuracy Calculation ---
if __name__ == "__main__":
    model_path = "best_double_conv_layers_model.keras"  # Path to your trained model
    calculate_test_accuracy(model_path, test_data_dir)
    ##allmodels = os.listdir("saved_models_32_50epochs")
    ##for model in allmodels:
    ##    if model.endswith(".keras"):
    ##        model_path = "saved_models_32_50epochs/" + model
    ##        calculate_test_accuracy(model_path, test_data_dir)
