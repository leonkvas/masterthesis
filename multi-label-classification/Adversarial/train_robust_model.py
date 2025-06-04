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

# Fixed parameters
IMG_SIZE = (50, 250)
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

# Define the vocabulary: digits 0-9 and uppercase A-Z
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# Directory paths
train_dir = "data/train2"
val_dir = "data/val2"
adversarial_dir = "data/train_2_adversarial_examples"

# Calculate max captcha length
train_files = [os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg'))]
val_files = [os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg'))]
max_captcha_len = max(max(len(name) for name in train_files), max(len(name) for name in val_files))
print(f"Detected max CAPTCHA length: {max_captcha_len}")

# --- Data Loading Functions ---
def load_image_and_label(file_path):
    file_path_str = file_path.numpy().decode('utf-8')
    
    # Load and preprocess image
    img = tf.io.read_file(file_path_str)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize
    
    # Extract label from filename
    label_str = os.path.splitext(os.path.basename(file_path_str))[0]
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
    return img, np.array(seq_padded, dtype=np.int32)

def process_path(file_path):
    img, label = tf.py_function(func=load_image_and_label, inp=[file_path], Tout=[tf.float32, tf.int32])
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
def create_datasets():
    # Training dataset (original)
    train_file_pattern = os.path.join(train_dir, "*.jpg")
    train_files_ds = tf.data.Dataset.list_files(train_file_pattern, shuffle=True)
    train_ds = train_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(
        lambda x, y: (augment_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Adversarial dataset - recursively find all image files
    adv_files = []
    for root, dirs, files in os.walk(adversarial_dir):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                adv_files.append(os.path.join(root, file))
    
    adv_files_ds = tf.data.Dataset.from_tensor_slices(adv_files)
    adv_ds = adv_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    adv_ds = adv_ds.map(
        lambda x, y: (augment_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Combine datasets
    combined_ds = train_ds.concatenate(adv_ds)
    combined_ds = combined_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_file_pattern = os.path.join(val_dir, "*.jpg")
    val_files_ds = tf.data.Dataset.list_files(val_file_pattern, shuffle=True)
    val_ds = val_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
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
def train_model():
    print(f"\n=== Training Robust Model (Double Conv Layers with Adversarial Training) ===\n")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Create datasets
    train_ds, val_ds = create_datasets()
    
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
        EarlyStopping(
            monitor='val_full_sequence_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            'best_robust_model2.keras',
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
    
    # Get the best accuracy from history
    best_accuracy = max(history.history['val_full_sequence_accuracy'])
    best_epoch = history.history['val_full_sequence_accuracy'].index(best_accuracy) + 1
    
    # Print results
    print(f"\nTraining Results:")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Full Sequence Accuracy: {seq_accuracy:.4f}")
    print(f"Best Full Sequence Accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"Model Parameters: {model.count_params():,}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_full_sequence_accuracy'], label='Validation')
    plt.plot(history.history['full_sequence_accuracy'], label='Training')
    plt.title('Sequence Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Validation')
    plt.plot(history.history['loss'], label='Training')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_robust.png')
    plt.close()
    
    print("\nTraining history plot saved as training_history_robust.png")
    print("Best model saved as best_robust_model2.keras")

# --- Main Execution ---
if __name__ == "__main__":
    train_model() 