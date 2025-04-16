import os
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters ---
IMG_SIZE = (50, 250)
BATCH_SIZE = 32

# Define the vocabulary: digits 0-9 and uppercase A-Z
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}  # Mapping: '0'->1, etc.
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
max_captcha_len = 7


def load_image_and_label(file_path):
    """
    Loads an image and its corresponding label.
    """
    # Convert EagerTensor to string
    file_path = file_path.numpy().decode('utf-8')

    # Load and preprocess image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1)  # Changed to 3 channels
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize

    # Extract label from the file name
    file_name = os.path.basename(file_path)
    label_str = os.path.splitext(file_name)[0]
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
    return img, seq_padded


def process_path(file_path):
    img, label = tf.py_function(func=load_image_and_label, inp=[file_path], Tout=[tf.float32, tf.int32])
    img.set_shape(IMG_SIZE + (1,))  # Changed to 3 channels
    label.set_shape([max_captcha_len])
    return img, label


def create_dataset(data_dir, batch_size=32):
    """
    Create a TensorFlow dataset from the image files in a directory.
    """
    file_pattern = os.path.join(data_dir, "*.jpg")
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_model(vocab_size):
    """
    Build a robust CNN CAPTCHA recognition model.
    """
    inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 1))  # Changed to 3 channels
    
    # Define data augmentation layers
    data_augmentation = tf.keras.Sequential([
        # layers.RandomRotation(factor=0.1),
        # layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        # layers.Lambda(lambda x: x + tf.random.uniform(tf.shape(x), minval=-0.05, maxval=0.05)),
        # layers.RandomShear(x_factor=0.1, y_factor=0.1),
        # layers.RandomBrightness(factor=0.2),
        # layers.RandomSharpness(factor=(0.4,0.5)),
    ])
    
    x = inputs  # Disabled data augmentation
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer: predict max_len characters, each with a probability distribution over vocab_size classes
    x = tf.keras.layers.Dense(max_captcha_len * vocab_size)(x)
    outputs = tf.keras.layers.Reshape((max_captcha_len, vocab_size))(x)
    outputs = tf.keras.layers.Activation('softmax')(outputs)
    
    return tf.keras.Model(inputs, outputs)


@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def hyperparameter_tuning(models_dir, train_dir, val_dir, test_dir, epochs_list, batch_sizes):
    """
    Perform hyperparameter tuning with different epochs and batch sizes.
    """
    results = []

    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f"\nTraining model with epochs={epochs}, batch_size={batch_size}")

            # Create new datasets with the current batch size
            train_ds = create_dataset(train_dir, batch_size=batch_size)
            val_ds = create_dataset(val_dir, batch_size=batch_size)
            test_ds = create_dataset(test_dir, batch_size=batch_size)

            # Create and compile the model
            model = create_model(vocab_size)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', full_sequence_accuracy]
            )

            # Train the model
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )

            # Save the trained model
            model_name = f"model_epochs{epochs}_batch{batch_size}.keras"
            model_path = os.path.join(models_dir, model_name)
            model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate on test dataset
            loss, accuracy, full_seq_acc = model.evaluate(test_ds)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Full Sequence Accuracy: {full_seq_acc:.4f}")

            # Save results
            results.append({
                "epochs": epochs,
                "batch_size": batch_size,
                "character_accuracy": accuracy,
                "full_sequence_accuracy": full_seq_acc,
                "model_path": model_path
            })

    return results


# --- Main Script ---
if __name__ == "__main__":
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "..", "data", "train")
    val_dir = os.path.join(base_dir, "..", "data", "val")
    test_dir = os.path.join(base_dir, "..", "data", "test")
    models_dir = os.path.join(base_dir, "saved_models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Hyperparameters to tune
    epochs_list = [32]  # Change as needed
    batch_sizes = [32]  # Change as needed

    # Perform hyperparameter tuning
    results = hyperparameter_tuning(models_dir, train_dir, val_dir, test_dir, epochs_list, batch_sizes)

    # Print structured results
    print("\nHyperparameter Tuning Results:")
    for result in results:
        print(f"Epochs: {result['epochs']}, Batch Size: {result['batch_size']}, "
              f"Character Accuracy: {result['character_accuracy']:.2f}%, "
              f"Full Sequence Accuracy: {result['full_sequence_accuracy']:.2f}%, "
              f"Model Path: {result['model_path']}")
