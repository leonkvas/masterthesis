import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd
import time
import itertools

# --- Parameters to Tune ---
LEARNING_RATES = [0.0001, 0.0005, 0.001]
BATCH_SIZES = [16, 32]
EPOCHS = [50, 75]

# Fixed parameters
IMG_SIZE = (50, 250)

# Define the vocabulary: digits 0-9 and uppercase A-Z
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# Directory paths
train_dir = "data/train2"
val_dir = "data/val2"

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
def create_datasets(batch_size):
    # Training dataset
    train_file_pattern = os.path.join(train_dir, "*.jpg")
    train_files_ds = tf.data.Dataset.list_files(train_file_pattern, shuffle=True)
    train_ds = train_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(
        lambda x, y: (augment_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_file_pattern = os.path.join(val_dir, "*.jpg")
    val_files_ds = tf.data.Dataset.list_files(val_file_pattern, shuffle=True)
    val_ds = val_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# --- Model Architectures ---

# Model 1: Double Conv Layers
def create_model_double_conv(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
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

# Model 2: With Residual Connection
def create_model_with_residual(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Add data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomSharpness(factor=(0.4, 0.8)),
    ])
    x = data_augmentation(inputs)
    
    # First and second blocks same as before
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third block with residual connection
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
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

# Model 3: Enhanced Augmentation
def create_model_enhanced_augmentation(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Enhanced data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomSharpness(factor=(0.4, 0.8)),
        layers.RandomRotation(factor=0.03),
        layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
    ])
    x = data_augmentation(inputs)
    
    # Architecture from Model 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third block with residual connection
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
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

# Model 4: Complete Model
def create_model_extra_layers(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Enhanced data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomSharpness(factor=(0.4, 0.8)),
        layers.RandomRotation(factor=0.03),
        layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
    ])
    x = data_augmentation(inputs)
    
    # First 3 blocks same as Model 3
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Add an extra convolutional layer group
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(384, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Additional dense layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(max_len * vocab_size)(x)
    outputs = layers.Reshape((max_len, vocab_size))(x)
    outputs = layers.Activation('softmax')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Define the models to test
model_configs = [
    {
        "name": "Double Conv Layers",
        "function": create_model_double_conv,
        "description": "Two conv layers per block for better feature extraction"
    },
    {
        "name": "With Residual Connection",
        "function": create_model_with_residual,
        "description": "Added residual connection in third block"
    },
    {
        "name": "Enhanced Augmentation",
        "function": create_model_enhanced_augmentation,
        "description": "Added rotation and zoom augmentations"
    }
]

# --- Training and Evaluation Function ---
def train_and_evaluate_model(model_fn, model_name, train_ds, val_ds, learning_rate, epochs, batch_size):
    print(f"\n=== Training Model: {model_name} ===\n")
    print(f"Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")
    
    # Build and compile model
    model = model_fn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', full_sequence_accuracy]
    )
    
    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_full_sequence_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            f'best_{model_name.replace(" ", "_").lower()}.keras',
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
        epochs=epochs,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    # Evaluate on validation data
    loss, accuracy, seq_accuracy = model.evaluate(val_ds)
    
    # Get the best accuracy from history
    best_accuracy = max(history.history['val_full_sequence_accuracy'])
    best_epoch = history.history['val_full_sequence_accuracy'].index(best_accuracy) + 1
    
    # Compile results
    results = {
        "Model Name": model_name,
        "Learning Rate": learning_rate,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Validation Loss": loss,
        "Validation Accuracy": accuracy,
        "Full Sequence Accuracy": seq_accuracy,
        "Best Sequence Accuracy": best_accuracy,
        "Best Epoch": best_epoch,
        "Training Time (s)": training_time,
        "Parameters": model.count_params()
    }
    
    return results, history

# --- Main Execution ---
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = "saved_models_hyperparameter_tuning_new"
    os.makedirs(results_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    histories = {}
    
    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(LEARNING_RATES, BATCH_SIZES, EPOCHS))
    
    # Train and evaluate each model with each hyperparameter combination
    for config in model_configs:
        model_name = config["name"]
        model_fn = config["function"]
        description = config["description"]
        
        print(f"\n{'='*50}")
        print(f"Starting evaluation of model: {model_name}")
        print(f"Description: {description}")
        print(f"{'='*50}\n")
        
        for lr, batch_size, epochs in hyperparameter_combinations:
            print(f"\nTesting combination: LR={lr}, Batch={batch_size}, Epochs={epochs}")
            
            # Create datasets with current batch size
            train_ds, val_ds = create_datasets(batch_size)
            
            # Train and evaluate
            results, history = train_and_evaluate_model(model_fn, model_name, train_ds, val_ds, lr, epochs, batch_size)
            all_results.append(results)
            histories[f"{model_name}_lr{lr}_batch{batch_size}_epochs{epochs}"] = history
            
            # Print results
            print(f"\nResults for {model_name} (LR={lr}, Batch={batch_size}, Epochs={epochs}):")
            print(f"Validation Loss: {results['Validation Loss']:.4f}")
            print(f"Validation Accuracy: {results['Validation Accuracy']:.4f}")
            print(f"Full Sequence Accuracy: {results['Full Sequence Accuracy']:.4f}")
            print(f"Best Full Sequence Accuracy: {results['Best Sequence Accuracy']:.4f} at epoch {results['Best Epoch']}")
            print(f"Model Parameters: {results['Parameters']:,}")
            print(f"Training Time: {results['Training Time (s)']:.2f} seconds")
    
    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, "hyperparameter_tuning_results.csv"), index=False)
    print("\nAll results saved to hyperparameter_tuning_results.csv")
    
    # Plot comparison of results
    plt.figure(figsize=(14, 8))
    
    # Plot sequence accuracy
    plt.subplot(1, 2, 1)
    plt.bar(results_df['Model Name'], results_df['Full Sequence Accuracy'], color='skyblue')
    plt.title('Full Sequence Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Plot parameter count
    plt.subplot(1, 2, 2)
    plt.bar(results_df['Model Name'], results_df['Parameters'] / 1000000, color='orange')
    plt.title('Model Size (Millions of Parameters)')
    plt.ylabel('Parameters (Millions)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hyperparameter_comparison.png"))
    plt.close()
    
    # Plot training histories
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy histories
    plt.subplot(2, 1, 1)
    for model_name, history in histories.items():
        plt.plot(history.history['val_full_sequence_accuracy'], label=model_name)
    
    plt.title('Validation Sequence Accuracy During Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Plot loss histories
    plt.subplot(2, 1, 2)
    for model_name, history in histories.items():
        plt.plot(history.history['val_loss'], label=model_name)
    
    plt.title('Validation Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_histories.png"))
    
    print("\nResults visualization saved as hyperparameter_comparison.png and training_histories.png") 