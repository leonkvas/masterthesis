import tensorflow as tf
import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Parameters (must match training) ---
IMG_SIZE = (50, 250)  # (height, width)
BATCH_SIZE = 16
CHANNELS = 1  # Grayscale images

# Attack parameters
FGSM_EPSILONS = [1, 2, 3]
GAUSSIAN_SIGMAS = [0.1, 0.2, 0.25]
IFGS_CONFIDENCE_TARGETS = [0.7, 0.8, 0.9]
BIM_EPSILONS = [0.1, 0.2, 0.5]
BIM_ITERATIONS = 40 # less iterations aswell which didnt give perfect results
SMOOTH_FGSM_PARAMS = [(0.5, 1.0), (0.5, 2.0), (1.0, 2.0), (1.5, 2.0), (2.0, 2.0)]  # (sigma, epsilon) pairs (based off the accuracy reduction heatmap.png)

num_samples = 50

# Model paths
SOURCE_MODEL_PATH = "best_double_conv_layers_model.keras"
OTHER_MODEL_PATHS = [
    "best_complete_model.keras",
    "best_enhanced_augmentation.keras",
    "best_with_residual_connection.keras"
]

# Vocabulary settings
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (index 0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
MAX_CAPTCHA_LEN = 7

# Directory for saving examples
ADV_EXAMPLES_DIR = "data/train_2_adversarial_examples" # change to test if you want to test on test set
#ADV_EXAMPLES_DIR = "multi-label-classification/Adversarial/transferability_examples" # change to train if you want to test on train set

# --- Helper functions ---
def load_and_preprocess_image(image_path):
    """Reads an image from 'image_path', resizes it to IMG_SIZE, and normalizes pixel values."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=CHANNELS)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0, 1]
    return img

def extract_label_from_filename(image_path):
    """Extracts the ground-truth label from the image filename."""
    base_name = os.path.basename(image_path)
    label_str = os.path.splitext(base_name)[0]
    return label_str

def label_to_sequence(label_str, max_len=MAX_CAPTCHA_LEN):
    """Converts a label string into a padded sequence of integers."""
    seq = [char_to_idx.get(char, 0) for char in label_str]
    seq_padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')[0]
    return np.array(seq_padded, dtype=np.int32)

def save_image(image_tensor, save_path):
    """Save a tensor image to disk."""
    if isinstance(image_tensor, tf.Tensor):
        image_tensor = image_tensor.numpy()
    
    # Ensure image is properly scaled for saving (0-255)
    image_array = (image_tensor * 255).astype(np.uint8)
    
    # Convert to PIL image and save
    if len(image_array.shape) == 3 and image_array.shape[-1] == 1:
        # Grayscale image with channel dimension
        image_array = image_array.squeeze()
    
    image = Image.fromarray(image_array)
    if image.mode != 'L':  # Convert to grayscale if needed
        image = image.convert('L')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

def create_fgsm_examples(model, test_files, test_dir):
    """Generate FGSM adversarial examples for each test file."""
    examples = []
    
    for epsilon in FGSM_EPSILONS:
        # Shuffle and select new subset for each epsilon
        random.shuffle(test_files)
        current_test_files = test_files[:num_samples]
        
        for file in current_test_files:
            image_path = os.path.join(test_dir, file)
            original_image = load_and_preprocess_image(image_path)
            original_label = extract_label_from_filename(file)
            label_sequence = label_to_sequence(original_label)
            
            # Skip files where original prediction is incorrect
            original_img_batch = tf.expand_dims(original_image, 0)
            original_preds = model.predict(original_img_batch, verbose=0)
            original_pred_indices = np.argmax(original_preds, axis=-1)[0]
            predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
            
            if predicted_original != original_label:
                print(f"Skipping {file}: original prediction incorrect")
                continue
            
            # Apply FGSM attack
            input_image = tf.convert_to_tensor(original_image)
            input_image = tf.expand_dims(input_image, 0)
            input_label = tf.expand_dims(label_sequence, 0)
            
            target_label = tf.roll(input_label, shift=1, axis=1)  # Shift all characters by one position
            adv_image = tf.identity(input_image)
            
            # PGD parameters for FGSM
            num_steps = 10
            alpha = 0.01
            
            for _ in range(num_steps):
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    
                    loss = 0
                    for i in range(prediction.shape[1]):
                        pos_pred = prediction[:, i, :]
                        pos_target = target_label[:, i]
                        
                        if pos_target[0] != 0:
                            pos_loss = tf.keras.losses.sparse_categorical_crossentropy(
                                pos_target, pos_pred, from_logits=False
                            )
                            loss += pos_loss
                    
                    non_padding_positions = tf.reduce_sum(tf.cast(target_label != 0, tf.float32))
                    loss = loss / (non_padding_positions + 1e-8)
                
                gradient = tape.gradient(loss, adv_image)
                adv_image = adv_image + alpha * tf.sign(gradient)
                adv_image = tf.clip_by_value(adv_image, 0, 1)
            
            perturbation = adv_image - input_image
            adversarial_image = original_image + epsilon * perturbation[0]
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
            
            # Check if attack is successful
            adv_img_batch = tf.expand_dims(adversarial_image, 0)
            adv_preds = model.predict(adv_img_batch, verbose=0)
            adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
            predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
            
            success = predicted_adv != original_label
            
            # Save the adversarial example
            example_id = f"{file.split('.')[0]}_fgsm_eps{epsilon:.1f}"
            save_dir = os.path.join(ADV_EXAMPLES_DIR, "fgsm", f"epsilon_{epsilon}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{example_id}.png")
            save_image(adversarial_image, save_path)
            
            # Record metadata
            examples.append({
                "original_file": file,
                "adversarial_file": f"{example_id}.png",
                "original_label": original_label,
                "predicted_original": predicted_original,
                "predicted_adversarial": predicted_adv,
                "attack_method": "fgsm",
                "attack_parameter": epsilon,
                "attack_success": success,
                "relative_path": os.path.join("fgsm", f"epsilon_{epsilon}", f"{example_id}.png")
            })
    
    return examples

def create_gaussian_noise_examples(model, test_files, test_dir):
    """Generate Gaussian noise adversarial examples for each test file."""
    examples = []
    
    for sigma in GAUSSIAN_SIGMAS:
        # Shuffle and select new subset for each sigma
        random.shuffle(test_files)
        current_test_files = test_files[:num_samples]
        
        for file in current_test_files:
            image_path = os.path.join(test_dir, file)
            original_image = load_and_preprocess_image(image_path)
            original_label = extract_label_from_filename(file)
            
            # Skip files where original prediction is incorrect
            original_img_batch = tf.expand_dims(original_image, 0)
            original_preds = model.predict(original_img_batch, verbose=0)
            original_pred_indices = np.argmax(original_preds, axis=-1)[0]
            predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
            
            if predicted_original != original_label:
                print(f"Skipping {file}: original prediction incorrect")
                continue
            
            # Apply Gaussian noise
            noise = tf.random.normal(shape=original_image.shape, mean=0.0, stddev=sigma, dtype=tf.float32)
            noisy_image = original_image + noise
            noisy_image = tf.clip_by_value(noisy_image, 0, 1)
            
            # Check if attack is successful
            noisy_img_batch = tf.expand_dims(noisy_image, 0)
            noisy_preds = model.predict(noisy_img_batch, verbose=0)
            noisy_pred_indices = np.argmax(noisy_preds, axis=-1)[0]
            predicted_noisy = ''.join([idx_to_char.get(idx, '') for idx in noisy_pred_indices if idx != 0])
            
            success = predicted_noisy != original_label
            
            # Save the adversarial example
            example_id = f"{file.split('.')[0]}_gaussian_sigma{sigma:.1f}"
            save_dir = os.path.join(ADV_EXAMPLES_DIR, "gaussian", f"sigma_{sigma}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{example_id}.png")
            save_image(noisy_image, save_path)
            
            # Record metadata
            examples.append({
                "original_file": file,
                "adversarial_file": f"{example_id}.png",
                "original_label": original_label,
                "predicted_original": predicted_original,
                "predicted_adversarial": predicted_noisy,
                "attack_method": "gaussian",
                "attack_parameter": sigma,
                "attack_success": success,
                "relative_path": os.path.join("gaussian", f"sigma_{sigma}", f"{example_id}.png")
            })
    
    return examples

def apply_gaussian_smoothing(image, sigma):
    """Apply Gaussian blur to the image using manual convolution."""
    # Convert image to tensor if it's not already
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # If image has no batch dimension, add it
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)  # Add channel dim
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)  # Add batch dim
    
    # Create Gaussian kernel
    kernel_size = 5  # Fixed kernel size
    x = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x, x)
    
    # Calculate 2D Gaussian kernel
    kernel = tf.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Add feature dimensions to kernel
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    
    # Apply convolution with Gaussian kernel
    smoothed_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    # Remove batch dimension if the original image didn't have it
    if len(tf.shape(image)) == 4 and image.shape[0] == 1:
        smoothed_image = smoothed_image[0]
    
    return smoothed_image

def create_smooth_fgsm_examples(model, test_files, test_dir):
    """Generate Smoothed FGSM adversarial examples for each test file."""
    examples = []
    
    for sigma, epsilon in SMOOTH_FGSM_PARAMS:
        # Shuffle and select new subset for each parameter pair
        random.shuffle(test_files)
        current_test_files = test_files[:num_samples]
        
        for file in current_test_files:
            image_path = os.path.join(test_dir, file)
            original_image = load_and_preprocess_image(image_path)
            original_label = extract_label_from_filename(file)
            label_sequence = label_to_sequence(original_label)
            
            # Skip files where original prediction is incorrect
            original_img_batch = tf.expand_dims(original_image, 0)
            original_preds = model.predict(original_img_batch, verbose=0)
            original_pred_indices = np.argmax(original_preds, axis=-1)[0]
            predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
            
            if predicted_original != original_label:
                print(f"Skipping {file}: original prediction incorrect")
                continue
            
            # First apply Gaussian smoothing
            smoothed_image = apply_gaussian_smoothing(original_image, sigma)
            if len(smoothed_image.shape) == 4:
                smoothed_image = smoothed_image[0]
            
            # Then apply FGSM on smoothed image
            input_image = tf.convert_to_tensor(smoothed_image)
            input_image = tf.expand_dims(input_image, 0)
            input_label = tf.expand_dims(label_sequence, 0)
            
            target_label = tf.roll(input_label, shift=1, axis=1)
            adv_image = tf.identity(input_image)
            
            num_steps = 10
            alpha = 0.01
            
            for _ in range(num_steps):
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    
                    loss = 0
                    for i in range(prediction.shape[1]):
                        pos_pred = prediction[:, i, :]
                        pos_target = target_label[:, i]
                        
                        if pos_target[0] != 0:
                            pos_loss = tf.keras.losses.sparse_categorical_crossentropy(
                                pos_target, pos_pred, from_logits=False
                            )
                            loss += pos_loss
                    
                    non_padding_positions = tf.reduce_sum(tf.cast(target_label != 0, tf.float32))
                    loss = loss / (non_padding_positions + 1e-8)
                
                gradient = tape.gradient(loss, adv_image)
                adv_image = adv_image + alpha * tf.sign(gradient)
                adv_image = tf.clip_by_value(adv_image, 0, 1)
            
            perturbation = adv_image - input_image
            
            # Apply perturbation to ORIGINAL image (not smoothed)
            perturbation = tf.squeeze(perturbation)
            if len(perturbation.shape) < len(original_image.shape):
                perturbation = tf.expand_dims(perturbation, -1)
            
            adversarial_image = original_image + epsilon * perturbation
            adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
            
            # Check if attack is successful
            adv_img_batch = tf.expand_dims(adversarial_image, 0)
            adv_preds = model.predict(adv_img_batch, verbose=0)
            adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
            predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
            
            success = predicted_adv != original_label
            
            # Save the adversarial example
            example_id = f"{file.split('.')[0]}_smooth_sigma{sigma:.1f}_eps{epsilon:.1f}"
            save_dir = os.path.join(ADV_EXAMPLES_DIR, "smooth_fgsm", f"sigma_{sigma}_epsilon_{epsilon}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{example_id}.png")
            save_image(adversarial_image, save_path)
            
            # Record metadata
            examples.append({
                "original_file": file,
                "adversarial_file": f"{example_id}.png",
                "original_label": original_label,
                "predicted_original": predicted_original,
                "predicted_adversarial": predicted_adv,
                "attack_method": "smooth_fgsm",
                "attack_parameters": {"sigma": sigma, "epsilon": epsilon},
                "attack_success": success,
                "relative_path": os.path.join("smooth_fgsm", f"sigma_{sigma}_epsilon_{epsilon}", f"{example_id}.png")
            })
    
    return examples

@tf.keras.utils.register_keras_serializable()
def full_sequence_accuracy(y_true, y_pred):
    """Custom metric to measure full sequence accuracy."""
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.reduce_all(tf.equal(y_true, y_pred_labels), axis=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def load_model_with_custom_objects(model_path):
    """Load model with custom metrics properly registered."""
    try:
        # Define custom objects dictionary for loading
        custom_objects = {
            'full_sequence_accuracy': full_sequence_accuracy
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model.trainable = False
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise e

def create_bim_examples(model, test_files, test_dir):
    """Generate BIM (Basic Iterative Method) adversarial examples for each test file."""
    examples = []
    RANDOM_START = True  # Whether to add random noise at the beginning
    for epsilon in BIM_EPSILONS:
        # Shuffle and select new subset for each epsilon
        random.shuffle(test_files)
        current_test_files = test_files[:num_samples]
        
        for file in current_test_files:
            image_path = os.path.join(test_dir, file)
            original_image = load_and_preprocess_image(image_path)
            original_label = extract_label_from_filename(file)
            label_sequence = label_to_sequence(original_label)
            
            # Skip files where original prediction is incorrect
            original_img_batch = tf.expand_dims(original_image, 0)
            original_preds = model.predict(original_img_batch, verbose=0)
            original_pred_indices = np.argmax(original_preds, axis=-1)[0]
            predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
            
            if predicted_original != original_label:
                print(f"Skipping {file}: original prediction incorrect")
                continue
            
            # Convert label to sequence and add batch dimension
            label_sequence_batch = tf.expand_dims(label_sequence, 0)
            
            # Get original loss
            prediction = model(original_img_batch)
            original_loss = tf.keras.losses.sparse_categorical_crossentropy(
                label_sequence_batch, prediction, from_logits=False
            )
            original_loss = tf.reduce_mean(original_loss)
            
            # Initialize adversarial example
            if RANDOM_START:
                # Add small random noise to start
                random_noise = tf.random.uniform(original_img_batch.shape, -epsilon/2, epsilon/2)
                adv_image = tf.clip_by_value(original_img_batch + random_noise, 0.0, 1.0)
            else:
                adv_image = tf.identity(original_img_batch)
            
            # Store the best adversarial example (with highest loss)
            best_adv_image = tf.identity(adv_image)
            best_loss = original_loss
            
            # Number of iterations for BIM
            num_iterations = BIM_ITERATIONS
            alpha = epsilon / num_iterations  # Step size
            
            # Track losses for visualization
            losses = []
            
            for i in range(num_iterations):
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    
                    # Calculate loss for all character positions at once
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        label_sequence_batch, prediction, from_logits=False
                    )
                    # Average across sequence positions
                    loss = tf.reduce_mean(loss)
                
                # Get gradients
                gradients = tape.gradient(loss, adv_image)
                
                # Update adversarial example using gradient sign (maximize loss)
                perturbation = alpha * tf.sign(gradients)
                adv_image = adv_image + perturbation
                
                # Clip the perturbation to maintain epsilon constraint
                delta = tf.clip_by_value(adv_image - original_img_batch, -epsilon, epsilon)
                adv_image = original_img_batch + delta
                
                # Clip to maintain valid pixel range [0, 1]
                adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)
                
                # Calculate current loss
                prediction = model(adv_image)
                current_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    label_sequence_batch, prediction, from_logits=False
                )
                current_loss = tf.reduce_mean(current_loss)
                losses.append(current_loss.numpy())
                
                # Track the best adversarial example
                if current_loss > best_loss:
                    best_adv_image = tf.identity(adv_image)
                    best_loss = current_loss
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i + 1}/{num_iterations}: Loss = {current_loss:.4f} (Original: {original_loss:.4f})")
            
            # Use the best adversarial example
            adversarial_image = best_adv_image[0]
            
            # Calculate the total change in loss
            loss_change = best_loss - original_loss
            
            # Check if attack is successful
            adv_img_batch = tf.expand_dims(adversarial_image, 0)
            adv_preds = model.predict(adv_img_batch, verbose=0)
            adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
            predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
            
            success = predicted_adv != original_label
            
            # Save the adversarial example
            example_id = f"{file.split('.')[0]}_bim_eps{epsilon:.3f}"
            save_dir = os.path.join(ADV_EXAMPLES_DIR, "bim", f"epsilon_{epsilon}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{example_id}.png")
            save_image(adversarial_image, save_path)
            
            # Record metadata
            examples.append({
                "original_file": file,
                "adversarial_file": f"{example_id}.png",
                "original_label": original_label,
                "predicted_original": predicted_original,
                "predicted_adversarial": predicted_adv,
                "attack_method": "bim",
                "attack_parameter": epsilon,
                "attack_iterations": BIM_ITERATIONS,
                "attack_success": success,
                "relative_path": os.path.join("bim", f"epsilon_{epsilon}", f"{example_id}.png")
            })
    
    return examples

def create_ifgs_examples(model, test_files, test_dir):
    """Generate IFGS (Iterative Fast Gradient Sign with target confidence) examples."""
    examples = []
    
    for confidence_target in IFGS_CONFIDENCE_TARGETS:
        # Shuffle and select new subset for each confidence target
        random.shuffle(test_files)
        current_test_files = test_files[:num_samples]
        
        for file in current_test_files:
            image_path = os.path.join(test_dir, file)
            original_image = load_and_preprocess_image(image_path)
            original_label = extract_label_from_filename(file)
            label_sequence = label_to_sequence(original_label)
            
            # Skip files where original prediction is incorrect
            original_img_batch = tf.expand_dims(original_image, 0)
            original_preds = model.predict(original_img_batch, verbose=0)
            original_pred_indices = np.argmax(original_preds, axis=-1)[0]
            predicted_original = ''.join([idx_to_char.get(idx, '') for idx in original_pred_indices if idx != 0])
            
            if predicted_original != original_label:
                print(f"Skipping {file}: original prediction incorrect")
                continue
            
            # Apply IFGS attack with confidence target
            input_image = tf.convert_to_tensor(original_image)
            input_image = tf.expand_dims(input_image, 0)
            label_batch = tf.expand_dims(label_sequence, 0)
            
            # Create target label by randomly changing a position
            target_label = tf.roll(label_batch, shift=1, axis=1)
            
            # Maximum iterations for IFGS
            max_iterations = 50
            epsilon = 0.01  # Initial step size
            
            # Initialize adversarial example with original image
            adv_image = tf.identity(input_image)
            iterations_used = 0
            max_confidence_achieved = 0.0
            
            # Iteratively apply gradient sign updates until target confidence is reached
            for iteration in range(max_iterations):
                iterations_used = iteration + 1
                
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    
                    # Calculate confidence in target prediction
                    confidences = []
                    for i in range(prediction.shape[1]):
                        pos_pred = prediction[:, i, :]
                        pos_target = target_label[:, i]
                        
                        if pos_target[0] != 0:
                            # Get confidence for target class
                            confidence = pos_pred[0, pos_target[0]]
                            confidences.append(confidence.numpy())
                    
                    # Average confidence across positions
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    max_confidence_achieved = max(max_confidence_achieved, avg_confidence)
                    
                    # If target confidence is achieved, break
                    if avg_confidence >= confidence_target:
                        break
                    
                    # Calculate loss toward target label
                    loss = 0
                    for i in range(prediction.shape[1]):
                        pos_pred = prediction[:, i, :]
                        pos_target = target_label[:, i]
                        
                        if pos_target[0] != 0:
                            pos_loss = tf.keras.losses.sparse_categorical_crossentropy(
                                pos_target, pos_pred, from_logits=False
                            )
                            loss += pos_loss
                    
                    non_padding_positions = tf.reduce_sum(tf.cast(target_label != 0, tf.float32))
                    loss = loss / (non_padding_positions + 1e-8)
                
                # Get gradients
                gradient = tape.gradient(loss, adv_image)
                
                # Update adversarial example
                adv_image = adv_image - epsilon * tf.sign(gradient)  # Note: negative for targeted attack
                
                # Ensure valid pixel range
                adv_image = tf.clip_by_value(adv_image, 0, 1)
                
                # Adaptive step size (reduce if oscillating)
                if iteration > 0 and iteration % 10 == 0:
                    epsilon = max(epsilon * 0.9, 0.001)  # Reduce step size but maintain minimum
            
            # Extract final adversarial example
            adversarial_image = adv_image[0]
            
            # Check if attack is successful
            adv_img_batch = tf.expand_dims(adversarial_image, 0)
            adv_preds = model.predict(adv_img_batch, verbose=0)
            adv_pred_indices = np.argmax(adv_preds, axis=-1)[0]
            predicted_adv = ''.join([idx_to_char.get(idx, '') for idx in adv_pred_indices if idx != 0])
            
            success = predicted_adv != original_label
            
            # Save the adversarial example
            example_id = f"{file.split('.')[0]}_ifgs_conf{confidence_target:.1f}"
            save_dir = os.path.join(ADV_EXAMPLES_DIR, "ifgs", f"confidence_{confidence_target}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{example_id}.png")
            save_image(adversarial_image, save_path)
            
            # Record metadata
            examples.append({
                "original_file": file,
                "adversarial_file": f"{example_id}.png",
                "original_label": original_label,
                "predicted_original": predicted_original,
                "predicted_adversarial": predicted_adv,
                "attack_method": "ifgs",
                "attack_parameter": confidence_target,
                "iterations_used": iterations_used,
                "max_confidence": float(max_confidence_achieved),
                "attack_success": success,
                "relative_path": os.path.join("ifgs", f"confidence_{confidence_target}", f"{example_id}.png")
            })
    
    return examples

def main():
    # Create main directory for adversarial examples
    os.makedirs(ADV_EXAMPLES_DIR, exist_ok=True)
    
    # Load source model
    print(f"Loading source model: {SOURCE_MODEL_PATH}")
    model = load_model_with_custom_objects(SOURCE_MODEL_PATH)
    
    # directory
    test_dir = "data/train2"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    print(f"Found {len(test_files)} test files")
    
    # Use a subset of test files for efficiency
    num_samples = 20
    # randomize the test files
    #random.shuffle(test_files)
    #test_files = test_files[:num_samples]
    print(f"Using {len(test_files)} test files for generating examples")
    
    all_examples = []

    # Generate BIM examples
    print("Generating BIM examples...")
    bim_examples = create_bim_examples(model, test_files, test_dir)
    all_examples.extend(bim_examples)
    print(f"Created {len(bim_examples)} BIM examples")
    
    # Generate FGSM examples
    print("Generating FGSM examples...")
    fgsm_examples = create_fgsm_examples(model, test_files, test_dir)
    all_examples.extend(fgsm_examples)
    print(f"Created {len(fgsm_examples)} FGSM examples")
    
    # Generate Gaussian noise examples
    print("Generating Gaussian noise examples...")
    gaussian_examples = create_gaussian_noise_examples(model, test_files, test_dir)
    all_examples.extend(gaussian_examples)
    print(f"Created {len(gaussian_examples)} Gaussian noise examples")
    
    # Generate IFGS examples
    print("Generating IFGS examples...")
    ifgs_examples = create_ifgs_examples(model, test_files, test_dir)
    all_examples.extend(ifgs_examples)
    print(f"Created {len(ifgs_examples)} IFGS examples")
    
    # Generate Smoothed FGSM examples
    print("Generating Smoothed FGSM examples...")
    smooth_examples = create_smooth_fgsm_examples(model, test_files, test_dir)
    all_examples.extend(smooth_examples)
    print(f"Created {len(smooth_examples)} Smoothed FGSM examples")
    
    # Save metadata to JSON
    metadata_path = os.path.join(ADV_EXAMPLES_DIR, "examples_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Also save a summary CSV
    csv_path = os.path.join(ADV_EXAMPLES_DIR, "examples_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["attack_method", "attack_parameter", "total_examples", "successful_attacks", "success_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # FGSM summary
        for epsilon in FGSM_EPSILONS:
            examples = [ex for ex in fgsm_examples if ex["attack_parameter"] == epsilon]
            successful = [ex for ex in examples if ex["attack_success"]]
            writer.writerow({
                "attack_method": "fgsm",
                "attack_parameter": f"epsilon={epsilon}",
                "total_examples": len(examples),
                "successful_attacks": len(successful),
                "success_rate": f"{len(successful)/len(examples):.2%}" if examples else "N/A"
            })
        
        # Gaussian noise summary
        for sigma in GAUSSIAN_SIGMAS:
            examples = [ex for ex in gaussian_examples if ex["attack_parameter"] == sigma]
            successful = [ex for ex in examples if ex["attack_success"]]
            writer.writerow({
                "attack_method": "gaussian",
                "attack_parameter": f"sigma={sigma}",
                "total_examples": len(examples),
                "successful_attacks": len(successful),
                "success_rate": f"{len(successful)/len(examples):.2%}" if examples else "N/A"
            })
        
        # BIM summary
        for epsilon in BIM_EPSILONS:
            examples = [ex for ex in bim_examples if ex["attack_parameter"] == epsilon]
            successful = [ex for ex in examples if ex["attack_success"]]
            writer.writerow({
                "attack_method": "bim",
                "attack_parameter": f"epsilon={epsilon}",
                "total_examples": len(examples),
                "successful_attacks": len(successful),
                "success_rate": f"{len(successful)/len(examples):.2%}" if examples else "N/A"
            })
        
        # IFGS summary
        for confidence in IFGS_CONFIDENCE_TARGETS:
            examples = [ex for ex in ifgs_examples if ex["attack_parameter"] == confidence]
            successful = [ex for ex in examples if ex["attack_success"]]
            writer.writerow({
                "attack_method": "ifgs",
                "attack_parameter": f"confidence={confidence}",
                "total_examples": len(examples),
                "successful_attacks": len(successful),
                "success_rate": f"{len(successful)/len(examples):.2%}" if examples else "N/A"
            })
        
        # Smoothed FGSM summary
        for sigma, epsilon in SMOOTH_FGSM_PARAMS:
            examples = [ex for ex in smooth_examples if ex["attack_parameters"]["sigma"] == sigma and 
                        ex["attack_parameters"]["epsilon"] == epsilon]
            successful = [ex for ex in examples if ex["attack_success"]]
            writer.writerow({
                "attack_method": "smooth_fgsm",
                "attack_parameter": f"sigma={sigma}, epsilon={epsilon}",
                "total_examples": len(examples),
                "successful_attacks": len(successful),
                "success_rate": f"{len(successful)/len(examples):.2%}" if examples else "N/A"
            })
    
    print(f"\nGenerated {len(all_examples)} adversarial examples")
    print(f"Metadata saved to {metadata_path}")
    print(f"Summary saved to {csv_path}")
    print("\nNext, use test_transferability.py to evaluate these examples against other models")

if __name__ == "__main__":
    main() 