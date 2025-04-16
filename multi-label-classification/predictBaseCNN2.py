import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Constants ---
IMG_SIZE = (50, 250)  # As defined in your original training setup
BATCH_SIZE = 32

# Test directory (contains the 50 test images moved earlier)
test_data_dir = "data/test2"

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


# Inside calculate_test_accuracy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def calculate_test_accuracy(model_path, test_data_dir):
    """
    Loads a pre-trained model and calculates prediction accuracy on the test dataset.
    Tracks and visualizes misclassified characters and their replacements.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Create the test dataset
    test_ds = create_test_dataset(test_data_dir)

    # Initialize counters for error analysis and metrics
    correct_predictions = 0
    total_predictions = 0
    error_counter = Counter()  # Track misclassifications as ('True', 'Predicted'): count
    label_counter = Counter()  # Track how often each true character appears

    for images, true_labels in test_ds:
        # Obtain predictions
        predictions = model.predict(images)  # Shape: (batch_size, max_captcha_len, vocab_size)

        # Convert predictions to indices
        pred_indices = tf.argmax(predictions, axis=-1).numpy()  # Shape: (batch_size, max_captcha_len)

        # Iterate through each prediction and ground truth in the batch
        for pred, true in zip(pred_indices, true_labels.numpy()):
            # Remove any padding (index 0) and decode to the actual label
            pred_decoded = [idx_to_char[idx] for idx in pred if idx != 0]
            true_decoded = [idx_to_char[idx] for idx in true if idx != 0]

            # Compare full label sequence
            if ''.join(pred_decoded) == ''.join(true_decoded):
                correct_predictions += 1
            else:
                print(f"Predicted: {''.join(pred_decoded)}, True: {''.join(true_decoded)}")

            # Track per-character misclassifications
            for true_char, pred_char in zip(true_decoded, pred_decoded):
                label_counter[true_char] += 1  # Count each occurrence of the true character
                if true_char != pred_char:
                    error_counter[(true_char, pred_char)] += 1

            total_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Test Accuracy: {accuracy * 100:.2f}% on {total_predictions} samples.")

    # Most frequent misclassifications
    print("\nMost Frequent Misclassifications:")
    for (true_char, pred_char), count in error_counter.most_common(10):  # Top 10 errors
        print(f"  '{true_char}' misclassified as '{pred_char}': {count} times.")


@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# --- Run the Accuracy Calculation ---
if __name__ == "__main__":
    #model_path = "saved_models_32_50epochs/best_double_conv_layers.keras"  # Path to your trained model
    #calculate_test_accuracy(model_path, test_data_dir)
    allmodels = os.listdir("saved_models_32_50epochs")
    for model in allmodels:
        if model.endswith(".keras"):
            model_path = "saved_models_32_50epochs/" + model
            calculate_test_accuracy(model_path, test_data_dir)
