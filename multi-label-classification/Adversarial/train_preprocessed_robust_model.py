import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from datetime import datetime
import json
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

# Fixed parameters
IMG_SIZE = (50, 250)
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

# Transformation parameters
BLOCK_SIZE = 8  # For block-wise transformations
JPEG_QUALITY = 75  # For JPEG compression
BIT_DEPTH = 4  # For bit-depth reduction
TOTAL_VARIANCE_WEIGHT = 0.1  # For TV minimization
GAUSSIAN_SIGMA = 1.0  # For Gaussian smoothing

# Define the vocabulary: digits 0-9 and uppercase A-Z
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# Directory paths
train_dir = "data/train2"
val_dir = "data/val2"
adversarial_dir = "data/train_2_adversarial_examples"
RESULTS_DIR = "multi-label-classification/Adversarial/preprocessing_results"

# Calculate max captcha length
train_files = [os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg'))]
val_files = [os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg'))]
max_captcha_len = max(max(len(name) for name in train_files), max(len(name) for name in val_files))
print(f"Detected max CAPTCHA length: {max_captcha_len}")

# --- Input Transformation Functions ---
def block_shuffle_transform(image, block_size=BLOCK_SIZE):
    """Apply block-wise shuffling transformation."""
    h, w = image.shape[:2]
    blocks = []
    positions = []
    
    # Calculate number of blocks in each dimension
    h_blocks = (h + block_size - 1) // block_size
    w_blocks = (w + block_size - 1) // block_size
    
    # Create padded image
    padded_h = h_blocks * block_size
    padded_w = w_blocks * block_size
    padded = np.zeros((padded_h, padded_w, image.shape[2]), dtype=image.dtype)
    padded[:h, :w] = image
    
    # Divide image into blocks
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            blocks.append(block)
            positions.append((i, j))
    
    # Shuffle blocks
    random.shuffle(blocks)
    
    # Reconstruct image
    shuffled = np.zeros_like(padded)
    for (i, j), block in zip(positions, blocks):
        shuffled[i:i+block_size, j:j+block_size] = block
    
    # Crop back to original size
    return shuffled[:h, :w]

def jpeg_compression_transform(image, quality=JPEG_QUALITY):
    """Apply JPEG compression transformation using only NumPy."""
    # Ensure image is in correct format
    if image.shape[-1] == 1:  # If grayscale
        img = np.squeeze(image)  # Remove channel dimension
    else:
        img = image
    
    # Simulate JPEG compression using NumPy
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Apply quantization (simulating JPEG compression)
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Scale quantization matrix based on quality
    scale = 1.0 if quality >= 50 else 50.0 / quality
    quantization_matrix = quantization_matrix * scale
    
    # Apply block-wise quantization
    h, w = img.shape[:2]
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:min(i+8, h), j:min(j+8, w)]
            if block.shape == (8, 8):
                block = np.round(block / quantization_matrix) * quantization_matrix
                img[i:i+8, j:j+8] = block
    
    # Convert back to float32 and normalize
    img = img.astype(np.float32) / 255.0
    
    # Restore channel dimension if needed
    if image.shape[-1] == 1:
        img = np.expand_dims(img, axis=-1)
    
    return img.astype(image.dtype)

def bit_depth_reduction_transform(image, bits=BIT_DEPTH):
    """Apply bit-depth reduction transformation using only NumPy."""
    # Ensure image is in correct format
    if image.shape[-1] == 1:  # If grayscale
        img = np.squeeze(image)  # Remove channel dimension
    else:
        img = image
    
    # Calculate maximum value for the given bit depth
    max_val = 2**bits - 1
    
    # Scale to the new bit depth
    img = np.round(img * max_val) / max_val
    
    # Ensure values stay in [0, 1] range
    img = np.clip(img, 0.0, 1.0)
    
    # Restore channel dimension if needed
    if image.shape[-1] == 1:
        img = np.expand_dims(img, axis=-1)
    
    return img.astype(image.dtype)

def total_variance_minimization_transform(image, weight=TOTAL_VARIANCE_WEIGHT):
    """Apply total variance minimization transformation using only NumPy."""
    # Ensure image is in correct format
    if image.shape[-1] == 1:  # If grayscale
        img = np.squeeze(image)  # Remove channel dimension
    else:
        img = image
    
    # Convert to float32 for calculations
    img = img.astype(np.float32)
    
    # Calculate gradients using NumPy's gradient
    grad_x = np.gradient(img, axis=1)
    grad_y = np.gradient(img, axis=0)
    
    # Calculate total variation
    tv = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply minimization
    result = img - weight * tv
    
    # Ensure values stay in [0, 1] range
    result = np.clip(result, 0.0, 1.0)
    
    # Restore channel dimension if needed
    if image.shape[-1] == 1:
        result = np.expand_dims(result, axis=-1)
    
    return result.astype(image.dtype)

def gaussian_smoothing_transform(image, sigma=GAUSSIAN_SIGMA):
    """Apply Gaussian smoothing transformation using only NumPy/Scipy."""
    # Ensure image is in correct format
    if image.shape[-1] == 1:  # If grayscale
        img = np.squeeze(image)  # Remove channel dimension
    else:
        img = image
    
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(img, sigma=sigma)
    
    # Ensure values stay in [0, 1] range
    smoothed = np.clip(smoothed, 0.0, 1.0)
    
    # Restore channel dimension if needed
    if image.shape[-1] == 1:
        smoothed = np.expand_dims(smoothed, axis=-1)
    
    return smoothed.astype(image.dtype)

def two_step_transform(image):
    """Apply two-step transformation (JPEG + TV minimization)."""
    # First step: JPEG compression
    step1 = jpeg_compression_transform(image)
    # Second step: TV minimization
    return total_variance_minimization_transform(step1)

# Dictionary of available transformations
TRANSFORMATIONS = {
    'gaussian_smoothing': gaussian_smoothing_transform,
    #'tv_minimization': total_variance_minimization_transform,
    #'bit_depth_reduction': bit_depth_reduction_transform,
    #'two_step': two_step_transform
}

# --- Data Loading Functions ---
def load_image_and_label(file_path, transform_func=None):
    file_path_str = file_path.numpy().decode('utf-8')
    
    # Load and preprocess image using tf.io
    img = tf.io.read_file(file_path_str)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize
    
    # Apply transformation if specified
    if transform_func:
        # Convert to numpy for transformation
        img_np = img.numpy()
        # Apply transformation
        transformed = transform_func(img_np)
        # Convert back to tensor
        img = tf.convert_to_tensor(transformed, dtype=tf.float32)
    
    # Extract label from filename
    label_str = os.path.splitext(os.path.basename(file_path_str))[0]
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
    return img, np.array(seq_padded, dtype=np.int32)

def process_path(file_path, transform_func=None):
    img, label = tf.py_function(
        func=lambda x: load_image_and_label(x, transform_func),
        inp=[file_path],
        Tout=[tf.float32, tf.int32]
    )
    img.set_shape(IMG_SIZE + (1,))
    label.set_shape([max_captcha_len])
    return img, label

def augment_image(image):
    # Apply random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Apply random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure values stay in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# --- Create Datasets ---
def create_datasets(transform_func=None):
    # Training dataset (original)
    train_file_pattern = os.path.join(train_dir, "*.jpg")
    train_files_ds = tf.data.Dataset.list_files(train_file_pattern, shuffle=True)
    train_ds = train_files_ds.map(
        lambda x: process_path(x, transform_func),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Adversarial dataset
    adv_files = []
    for root, dirs, files in os.walk(adversarial_dir):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                adv_files.append(os.path.join(root, file))
    
    adv_files_ds = tf.data.Dataset.from_tensor_slices(adv_files)
    adv_ds = adv_files_ds.map(
        lambda x: process_path(x, transform_func),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Combine datasets
    combined_ds = train_ds.concatenate(adv_ds)
    combined_ds = combined_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_file_pattern = os.path.join(val_dir, "*.jpg")
    val_files_ds = tf.data.Dataset.list_files(val_file_pattern, shuffle=True)
    val_ds = val_files_ds.map(
        lambda x: process_path(x, transform_func),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return combined_ds, val_ds

# --- Model Architecture ---
def create_model(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Add data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomSharpness(factor=(0.4, 0.8)),
    ])
    x = data_augmentation(inputs)
    
    # Double conv layers in each block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    
    x = layers.Dense(max_len * vocab_size)(x)
    outputs = layers.Reshape((max_len, vocab_size))(x)
    outputs = layers.Activation('softmax')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# --- Training Function ---
def train_model_with_transformation(transform_name, transform_func):
    print(f"\n=== Training Model with {transform_name} Transformation ===\n")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(RESULTS_DIR, f"{transform_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Clear any existing PIL/Tkinter objects
    import gc
    gc.collect()
    
    # Create datasets with transformation
    train_ds, val_ds = create_datasets(transform_func)
    
    # Build and compile model
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', full_sequence_accuracy]
    )
    
    # Define callbacks
    callbacks = [
        #EarlyStopping(
        #    monitor='val_full_sequence_accuracy',
        #    patience=10,
        #    restore_best_weights=True,
        #    verbose=1,
       #     mode='max'
       # ),
        ModelCheckpoint(
            os.path.join(model_dir, f"best_model.keras"),
            monitor='val_full_sequence_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    # Evaluate on validation data
    loss, accuracy, seq_accuracy = model.evaluate(val_ds)
    
    # Save results
    results = {
        "transform_name": transform_name,
        "timestamp": timestamp,
        "training_time": training_time,
        "final_loss": float(loss),
        "final_accuracy": float(accuracy),
        "final_sequence_accuracy": float(seq_accuracy),
        "best_accuracy": float(max(history.history['val_full_sequence_accuracy'])),
        "best_epoch": int(history.history['val_full_sequence_accuracy'].index(max(history.history['val_full_sequence_accuracy']))) + 1,
        "model_parameters": int(model.count_params())
    }
    
    # Save results to JSON
    with open(os.path.join(model_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_full_sequence_accuracy'], label='Validation')
    plt.plot(history.history['full_sequence_accuracy'], label='Training')
    plt.title('Sequence Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Validation')
    plt.plot(history.history['loss'], label='Training')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_history.png"))
    plt.close()
    
    # Clear any remaining PIL/Tkinter objects
    gc.collect()
    
    print(f"\nResults saved to {model_dir}")
    return results

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Train models with different transformations
    all_results = []
    for transform_name, transform_func in TRANSFORMATIONS.items():
        try:
            # Clear any existing PIL/Tkinter objects before each transformation
            import gc
            gc.collect()
            
            results = train_model_with_transformation(transform_name, transform_func)
            all_results.append(results)
        except Exception as e:
            print(f"Error training with {transform_name}: {e}")
    
    # Save summary of all results
    summary_path = os.path.join(RESULTS_DIR, "transformation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nTraining complete! Summary saved to", summary_path)

if __name__ == "__main__":
    main() 