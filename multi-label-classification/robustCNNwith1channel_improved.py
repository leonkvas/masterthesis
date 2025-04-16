import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# --- Parameters ---
IMG_SIZE = (50, 250)  # Adjust according to your images
BATCH_SIZE = 32
EPOCHS = 50  # Increased from 32

# Define the vocabulary: digits 0-9 and uppercase A-Z (adjust as needed)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}  # Mapping: '0'->1, etc.
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# --- Determine Maximum CAPTCHA Length from Filenames ---
train_dir = "data/train2"  # Folder containing training CAPTCHA images
val_dir = "data/val2"     # Folder containing validation CAPTCHA images

# Get file lists from both directories
train_files = [os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg'))]
val_files = [os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg'))]

# Find max length from both train and validation sets
max_captcha_len = max(max(len(name) for name in train_files), max(len(name) for name in val_files))
print(f"Detected max CAPTCHA length: {max_captcha_len}")

def plot_sample_images(dataset, num_images=5):
    """
    Plots a specified number of preprocessed images from the dataset.

    Args:
        dataset: A tf.data.Dataset instance returning `(image, label)` tuples.
        num_images: Number of images to display.
    """
    plt.figure(figsize=(15, 15))

    # Using the dataset, iterate to get individual images and labels
    for i, (images, labels) in enumerate(dataset.take(1)):  # Take 1 batch
        for j in range(num_images):
            if j >= len(images):
                break
            ax = plt.subplot(1, num_images, j + 1)  # Create subplots
            image = images[j]  # Extract a single image from the batch
            label = labels[j]
            plt.imshow(tf.keras.utils.array_to_img(image))
            decoded_label = ''.join([idx_to_char[idx] for idx in label.numpy() if idx != 0])  # Skip padding (index 0)
            plt.title(f"Label: {decoded_label}")
            plt.axis("off")
    plt.show()


# --- Helper functions ---
def load_image_and_label(file_path):
    file_path_str = file_path.numpy().decode('utf-8')

    # Load and preprocess image
    img = tf.io.read_file(file_path_str)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize

    # Extract label from filename (assumes filename is the ground-truth string)
    label_str = os.path.splitext(os.path.basename(file_path_str))[0]
    seq = [char_to_idx.get(char, 0) for char in label_str]
    # Pad sequence to the fixed max_captcha_len
    seq_padded = pad_sequences([seq], maxlen=max_captcha_len, padding='post', truncating='post')[0]
    return img, np.array(seq_padded, dtype=np.int32)


def process_path(file_path):
    img, label = tf.py_function(func=load_image_and_label, inp=[file_path], Tout=[tf.float32, tf.int32])
    img.set_shape(IMG_SIZE + (1,))
    label.set_shape([max_captcha_len])
    return img, label


# --- Data Augmentation Function ---
def augment_image(image):
    # Apply random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Apply random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure values stay in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# --- Create tf.data.Dataset ---
train_file_pattern = os.path.join(train_dir, "*.jpg")  # or ".jpg" as needed
val_file_pattern = os.path.join(val_dir, "*.jpg")      # or ".jpg" as needed

# Create training dataset with augmentation
train_files_ds = tf.data.Dataset.list_files(train_file_pattern, shuffle=True)
train_ds = train_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
# Apply augmentation only to images, not labels
train_ds = train_ds.map(
    lambda x, y: (augment_image(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
# Add repeat for more training examples
train_ds = train_ds.repeat(2)  # Double the effective dataset size
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create validation dataset
val_files_ds = tf.data.Dataset.list_files(val_file_pattern, shuffle=True)
val_ds = val_files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Example usage (assumes `train_ds` is your training dataset)
#plot_sample_images(train_ds)

def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# --- Build an Improved CNN Model ---
def create_model(input_shape=IMG_SIZE + (1,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)

    # Define data augmentation layers for training time only
    data_augmentation = tf.keras.Sequential([
        # More aggressive augmentation for problematic character cases (7/T, G/C, M/V, W/M, etc.)
        layers.RandomSharpness(factor=(0.4, 0.8)),
        # Add slight rotation to help with similar characters
        layers.RandomRotation(factor=0.03),
        # Small zoom can help with character proportions
        layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
    ])
    
    # Apply augmentation
    x = data_augmentation(inputs)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Third convolutional block with residual connection
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # Project shortcut to match dimension if needed
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    x = layers.add([x, shortcut])  # Residual connection
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Add a deeper layer specifically to distinguish common confusions
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(384, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Slightly reduced dropout for more learning
    
    # Add another dense layer to improve feature extraction
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer: predict max_len characters
    x = layers.Dense(max_len * vocab_size)(x)
    outputs = layers.Reshape((max_len, vocab_size))(x)
    outputs = layers.Activation('softmax')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    return model


# Create and compile the model with a slightly lower learning rate
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy', full_sequence_accuracy]
)

model.summary()

# --- Define Callbacks ---
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_full_sequence_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_captcha_model.keras',
    monitor='val_full_sequence_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# --- Train the Model ---
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping, model_checkpoint]
)

# --- Evaluate and Save the Model ---
loss, accuracy, full_sequence_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}, Validation Full Sequence Accuracy: {full_sequence_accuracy:.4f}")

# Save the final model
model.save("captcha_cnn_model_improved.keras")

# --- Plot Training History ---
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['full_sequence_accuracy'])
plt.plot(history.history['val_full_sequence_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val', 'Train Seq', 'Val Seq'], loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# --- Analyze Common Misclassifications ---
def analyze_misclassifications(model, dataset, idx_to_char):
    """
    Analyzes the most common types of misclassifications to understand model weaknesses.
    """
    misclassifications = {}
    total_samples = 0
    correct_samples = 0
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        pred_indices = tf.argmax(predictions, axis=-1).numpy()
        
        for i in range(len(labels)):
            true_indices = labels[i].numpy()
            pred_idx = pred_indices[i]
            
            # Get readable labels
            true_chars = [idx_to_char.get(idx, '') for idx in true_indices if idx != 0]
            pred_chars = [idx_to_char.get(idx, '') for idx in pred_idx if idx != 0]
            
            true_text = ''.join(true_chars)
            pred_text = ''.join(pred_chars)
            
            total_samples += 1
            if true_text == pred_text:
                correct_samples += 1
                continue
            
            # If prediction length doesn't match true length, skip detailed analysis
            if len(true_text) != len(pred_text):
                key = f"LENGTH_MISMATCH: {len(true_text)} vs {len(pred_text)}"
                misclassifications[key] = misclassifications.get(key, 0) + 1
                continue
            
            # Check character-by-character misclassifications
            for j in range(min(len(true_text), len(pred_text))):
                if j < len(true_text) and j < len(pred_text) and true_text[j] != pred_text[j]:
                    key = f"'{true_text[j]}' misclassified as '{pred_text[j]}'"
                    misclassifications[key] = misclassifications.get(key, 0) + 1
    
    # Sort by frequency
    sorted_misclass = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTest Accuracy: {correct_samples/total_samples:.2%} on {total_samples} samples.\n")
    print("Most Frequent Misclassifications:")
    for error, count in sorted_misclass[:15]:  # Show top 15
        print(f"  {error}: {count} times.")
    
    return sorted_misclass

# Run the analysis on validation dataset
print("\nAnalyzing common misclassifications...")
misclassifications = analyze_misclassifications(model, val_ds, idx_to_char)

# --- Predict on a single image ---
def predict_captcha(img_path):
    """
    Loads an image from img_path, uses the trained model to predict the CAPTCHA text,
    and returns the predicted string.
    """
    # Load and preprocess the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1)  # Changed to 1 channel
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize
    # Expand dimensions to create a batch of size 1
    img_batch = tf.expand_dims(img, axis=0)

    # Predict; expected output shape: (1, max_captcha_len, vocab_size)
    preds = model.predict(img_batch)

    # For each character position, pick the class with highest probability
    pred_indices = np.argmax(preds, axis=-1)[0]

    # Map indices to characters (skip any padding zeros)
    pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices if idx != 0]
    predicted_label = ''.join(pred_chars)
    return predicted_label 