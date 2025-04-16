import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Parameters ---
IMG_SIZE = (50, 250)  # Adjust according to your images
BATCH_SIZE = 32
EPOCHS = 16

# Define the vocabulary: digits 0-9 and uppercase A-Z (adjust as needed)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}  # Mapping: '0'->1, etc.
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}

# --- Determine Maximum CAPTCHA Length from Filenames ---
data_dir = "../getStringLabelByBoundingBoxes/labeled_captchas"  # Folder containing labeled CAPTCHA images
file_list = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
max_captcha_len = max(len(name) for name in file_list)
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
            plt.title(f"Label: {label.numpy()}")
            plt.axis("off")
    plt.show()


# --- Helper functions ---
def load_image_and_label(file_path):
    file_path_str = file_path.numpy().decode('utf-8')

    # Load and preprocess image
    img = tf.io.read_file(file_path_str)
    img = tf.image.decode_image(img, channels=3)
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
    img.set_shape(IMG_SIZE + (3,))
    label.set_shape([max_captcha_len])
    return img, label


# --- Create tf.data.Dataset ---
file_pattern = os.path.join(data_dir, "*.jpg")  # or ".jpg" as needed
files_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
dataset = files_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
# Now, since labels are already padded to [max_captcha_len], we can simply batch without further padding.
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- (Optional) Split into training and validation datasets ---
dataset = dataset.shuffle(1000)
total_batches = sum(1 for _ in dataset)
train_batches = int(total_batches * 0.7)
train_ds = dataset.take(train_batches)
# Example usage (assumes `train_ds` is your training dataset)
plot_sample_images(train_ds)
val_ds = dataset.skip(train_batches)

def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# --- Build a Custom CNN Model ---
def create_model(input_shape=IMG_SIZE + (3,), max_len=max_captcha_len, vocab_size=vocab_size):
    inputs = tf.keras.Input(shape=input_shape)

    # Define data augmentation layers
    #data_augmentation = tf.keras.Sequential([
        # layers.RandomRotation(factor=0.1),            # Randomly rotates images by up to 10%
        # layers.RandomZoom(height_factor=0.2, width_factor=0.2),  # Randomly zooms images by up to 20%
        # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Randomly shifts images by 10%
        # layers.Lambda(lambda x: x + tf.random.uniform(tf.shape(x), minval=-0.05, maxval=0.05)),  # Adds random noise
        #layers.RandomShear(x_factor=0.1, y_factor=0.1),  # Random shear
        # layers.RandomBrightness(factor=0.2),  # Random brightness/exposure
        #layers.RandomSharpness(factor=(0.4,0.5)),
    #])
    #x = data_augmentation(inputs)
    x = inputs
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)  # Dropout for regularization
    # Output layer: predict max_len characters, each with a probability distribution over vocab_size classes.
    x = layers.Dense(max_len * vocab_size)(x)
    outputs = layers.Reshape((max_len, vocab_size))(x)
    outputs = layers.Activation('softmax')(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model


model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', full_sequence_accuracy])
model.summary()

# --- Train the Model ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- Evaluate and Save the Model ---
loss, accuracy, full_sequence_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}, Validation Full Sequence Accuracy: {full_sequence_accuracy:.4f}")
model.save("captcha_cnn_model.keras")
#model.save("captcha_cnn_model_augmented.keras")

def predict_captcha(img_path):
    """
    Loads an image from img_path, uses the trained model to predict the CAPTCHA text,
    and returns the predicted string.
    """
    # Load and preprocess the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize
    # Expand dimensions to create a batch of size 1
    img_batch = tf.expand_dims(img, axis=0)

    # Load the trained model (ensure the path is correct)
    #model = tf.keras.models.load_model("captcha_cnn_model98%.keras")

    # Predict; expected output shape: (1, max_captcha_len, vocab_size)
    preds = model.predict(img_batch)

    # For each character position, pick the class with highest probability
    pred_indices = np.argmax(preds, axis=-1)[0]

    # Map indices to characters (skip any padding zeros)
    pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices if idx != 0]
    predicted_label = ''.join(pred_chars)
    return predicted_label

#label = predict_captcha("../Scraping/scrapedCaptchas/queueit_new_67.png")
#print(label)
