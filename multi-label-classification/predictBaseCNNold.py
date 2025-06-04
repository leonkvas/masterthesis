import time

import tensorflow as tf
import numpy as np
import os

# --- Parameters (must match training) ---
IMG_SIZE = (50, 250)
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
# Set max captcha length to the value used in training
max_captcha_len = 7  # Adjust if different


def load_and_preprocess_image(img_path):
    """
    Reads an image from the given path, resizes it, and normalizes pixel values.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1]
    #img.set_shape(IMG_SIZE + (3,))
    return img


def predict_captcha(img_path):
    """
    Loads an image from img_path, uses the trained model to predict the CAPTCHA text,
    and returns the predicted string.
    """
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)
    # Expand dimensions to create a batch of size 1
    img_batch = tf.expand_dims(img, axis=0)

    # Load the trained model (ensure the path is correct)
    model = tf.keras.models.load_model("captcha_cnn_model.keras")

    # Predict; expected output shape: (1, max_captcha_len, vocab_size)
    preds = model.predict(img_batch)

    # For each character position, pick the class with highest probability
    pred_indices = np.argmax(preds, axis=-1)[0]

    # Map indices to characters (skip any padding zeros)
    pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices if idx != 0]
    predicted_label = ''.join(pred_chars)
    return predicted_label

@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# --- Example usage ---
if __name__ == "__main__":
    # Path to the new CAPTCHA image
    new_img_path = "../Scraping/scrapedCaptchas/queueit_new_92.png"  # Replace with your image file path
    predicted_label = predict_captcha(new_img_path)
    print("Predicted CAPTCHA:", predicted_label)
    #time.sleep(222)
    # predict random images of labelled captchas and check if the prediction is correct
    # Loop through the directory and predict each image
    data_dir = "../getStringLabelByBoundingBoxes/test"  # Folder containing labeled CAPTCHA images
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(data_dir, filename)
            predicted_label = predict_captcha(img_path)
            print(f"Predicted CAPTCHA for {filename}: {predicted_label}")
