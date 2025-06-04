import tensorflow as tf
from tensorflow.keras import layers, models
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
import time
from datetime import datetime

# Define the character set
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_CHARS = len(CHARACTERS)
NUM_DIGITS = 10  # 0-9

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image with our best preprocessing steps"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float64) / 255.0  # Convert to float64
    
    # Add channel dimension
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Apply our best preprocessing steps
    # 1. Contrast enhancement
    img_array = tf.image.adjust_contrast(img_array, 2.0)
    
    # 2. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float64)
    kernel = np.expand_dims(kernel, axis=-1)
    kernel = np.expand_dims(kernel, axis=-1)
    
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.nn.conv2d(img_array, kernel, strides=[1,1,1,1], padding='SAME')
    img_array = img_array[0]
    
    # 3. Normalization
    img_array = (img_array - 0.5) / 0.5
    
    return img_array

def get_targets(label):
    """Extract number positions (1,4,7) and other positions"""
    # For 6-char captchas: positions 1,4 are numbers
    # For 7-char captchas: positions 1,4,7 are numbers
    number_positions = [label[0], label[3]]
    if len(label) == 7:
        number_positions.append(label[6])
    
    other_positions = [c for i, c in enumerate(label) if i not in [0, 3, 6]]
    
    # Convert to indices
    number_indices = [CHARACTERS.index(c) for c in number_positions]
    other_indices = [CHARACTERS.index(c) for c in other_positions]
    
    return number_indices, other_indices

class CaptchaDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.glob('*.jpg'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = load_and_preprocess_image(img_path)
        
        # Extract label from filename
        label = img_path.stem
        
        # Validate label length
        if len(label) not in [6, 7]:
            raise ValueError(f"Invalid label length in {img_path.name}. Expected 6 or 7 characters, got {len(label)}")
        
        number_targets, other_targets = get_targets(label)
        
        # Pad to fixed size
        number_targets = number_targets + [0] * (3 - len(number_targets))  # Pad to length 3
        other_targets = other_targets + [0] * (4 - len(other_targets))     # Pad to length 4
        
        # Convert to tensors with fixed size
        number_targets = tf.convert_to_tensor(number_targets, dtype=tf.int64)
        other_targets = tf.convert_to_tensor(other_targets, dtype=tf.int64)
        
        return img, (number_targets, other_targets)

def create_tf_dataset(dataset):
    """Convert CaptchaDataset to tf.data.Dataset"""
    def generator():
        for i in range(len(dataset)):
            img, (number_targets, other_targets) = dataset[i]
            yield img, (number_targets, other_targets)
    
    # Create dataset from generator
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(3,), dtype=tf.int64),
                tf.TensorSpec(shape=(4,), dtype=tf.int64)
            )
        )
    )
    
    # Get the original dataset size before repeating
    dataset_size = len(dataset)
    
    # Repeat the dataset indefinitely and batch
    tf_dataset = tf_dataset.repeat().batch(32).prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset, dataset_size

def create_model():
    """Create our best model architecture with separate heads for numbers and other characters"""
    input_layer = layers.Input(shape=(128, 128, 1))
    
    # Add data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomSharpness(factor=(0.4, 0.8)),
    ])
    x = data_augmentation(input_layer)
    
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
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    
    # Separate heads for number positions and other positions
    number_head = layers.Dense(256, activation='relu')(x)
    number_head = layers.Dropout(0.45)(number_head)
    number_output = layers.Dense(3 * NUM_DIGITS, name='number_output')(number_head)
    number_output = layers.Reshape((3, NUM_DIGITS))(number_output)
    number_output = layers.Activation('softmax')(number_output)
    
    other_head = layers.Dense(256, activation='relu')(x)
    other_head = layers.Dropout(0.45)(other_head)
    other_output = layers.Dense(4 * NUM_CHARS, name='other_output')(other_head)
    other_output = layers.Reshape((4, NUM_CHARS))(other_output)
    other_output = layers.Activation('softmax')(other_output)
    
    # Create model with multiple outputs (list format)
    model = models.Model(inputs=input_layer, outputs=[number_output, other_output])
    return model

def custom_loss(y_true, y_pred):
    """Custom loss function to handle variable length sequences"""
    # Calculate losses
    number_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true[0], y_pred[0])
    other_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true[1], y_pred[1])
    
    return number_loss + other_loss

def full_sequence_accuracy(y_true, y_pred):
    """Custom metric to calculate full sequence accuracy"""
    # Get true values
    number_true = y_true[0]
    other_true = y_true[1]
    
    # Get predictions
    number_pred = y_pred[0]
    other_pred = y_pred[1]
    
    # Get predicted classes
    number_pred_classes = tf.argmax(number_pred, axis=-1)
    other_pred_classes = tf.argmax(other_pred, axis=-1)
    
    # Convert to same type
    number_true = tf.cast(number_true, tf.int64)
    other_true = tf.cast(other_true, tf.int64)
    
    # Check if all number positions are correct
    number_correct = tf.reduce_all(
        tf.equal(number_pred_classes, number_true),
        axis=-1
    )
    
    # Check if all other positions are correct
    other_correct = tf.reduce_all(
        tf.equal(other_pred_classes, other_true),
        axis=-1
    )
    
    # Both number and other positions must be correct for full sequence accuracy
    full_sequence_correct = tf.logical_and(number_correct, other_correct)
    
    return tf.reduce_mean(tf.cast(full_sequence_correct, tf.float32))

def train_model(model, train_dataset, val_dataset, num_epochs, save_dir):
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'model_checkpoints/model_positional_tf_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original dataset size before unpacking for steps calculation
    train_data, train_size = train_dataset
    val_data, val_size = val_dataset
    
    steps_per_epoch = train_size // 32
    validation_steps = val_size // 32
    
    # Create a separate callback for full sequence accuracy
    class FullSequenceCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            full_seq_acc = 0
            batches = 0
            
            # Evaluate on validation data
            for x_batch, y_batch in val_data.take(validation_steps):
                y_pred = self.model.predict(x_batch, verbose=0)
                batch_acc = full_sequence_accuracy(y_batch, y_pred)
                full_seq_acc += batch_acc
                batches += 1
            
            # Calculate average accuracy
            if batches > 0:
                full_seq_acc /= batches
                logs['full_sequence_accuracy'] = full_seq_acc
                logs['val_full_sequence_accuracy'] = full_seq_acc
                print(f"\nEpoch {epoch+1}: Full Sequence Accuracy: {full_seq_acc:.4f}")
    
    # Compile model with metrics for each output only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
        metrics=['accuracy', 'accuracy']  # Exactly one metric per output
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.keras'),
            save_best_only=True,
            monitor='val_full_sequence_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_full_sequence_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        FullSequenceCallback()
    ]
    
    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Save training configuration
    config = {
        'model_architecture': model.to_json(),
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'num_epochs': num_epochs,
        'timestamp': timestamp,
        'description': 'TensorFlow model with separate heads for number positions (1,4,7) and other positions'
    }
    
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return history

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create datasets
    train_dataset = CaptchaDataset('data/train2')
    val_dataset = CaptchaDataset('data/val2')
    
    # Convert to tf.data.Dataset and get sizes
    train_tf_dataset = create_tf_dataset(train_dataset)
    val_tf_dataset = create_tf_dataset(val_dataset)
    
    # Create model
    model = create_model()
    
    # Train model
    num_epochs = 50
    history = train_model(model, train_tf_dataset, val_tf_dataset, num_epochs, 'model_checkpoints')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy for outputs and full sequence
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_1_accuracy'], label='Number Position Accuracy')
    plt.plot(history.history['output_2_accuracy'], label='Other Position Accuracy')
    plt.plot(history.history['full_sequence_accuracy'], label='Full Sequence Accuracy')
    plt.plot(history.history['val_output_1_accuracy'], label='Val Number Position Accuracy')
    plt.plot(history.history['val_output_2_accuracy'], label='Val Other Position Accuracy')
    plt.plot(history.history['val_full_sequence_accuracy'], label='Val Full Sequence Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    main() 