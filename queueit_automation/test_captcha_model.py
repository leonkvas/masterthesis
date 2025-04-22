import os
import sys
import argparse
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

def decode_base64_to_image(base64_string):
    """Decode a base64 string to an image"""
    # Add padding if needed
    padding = len(base64_string) % 4
    if padding:
        base64_string += '=' * (4 - padding)
    
    try:
        image_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(image_data))
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def preprocess_image(img, target_width=128, target_height=32):
    """Preprocess the image for the model"""
    try:
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to expected dimensions
        img = img.resize((target_width, target_height))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Reshape for model input (add batch dimension and channel dimension)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_captcha(model, img_array, char_set="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
    """Run prediction with the model"""
    try:
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Process prediction to get text
        result = ""
        for char_probs in prediction[0]:
            if len(char_probs.shape) > 0:  # If it's a vector of probabilities
                char_idx = np.argmax(char_probs)
                if char_idx < len(char_set):  # Ensure index is valid
                    result += char_set[char_idx]
        
        return result
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test captcha solving with Keras model')
    parser.add_argument('--model', type=str, default='best_double_conv_layers_model.keras',
                        help='Path to the Keras model file')
    parser.add_argument('--base64', type=str, default='',
                        help='Base64 encoded image string')
    parser.add_argument('--image', type=str, default='',
                        help='Path to image file')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return 1
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print(f"Model loaded successfully: {model.summary()}")
    
    # Get the image
    img = None
    if args.base64:
        print("Decoding base64 image...")
        img = decode_base64_to_image(args.base64)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found")
            return 1
        print(f"Opening image from {args.image}...")
        img = Image.open(args.image)
    else:
        print("Error: Please provide either a base64 string or an image file")
        return 1
    
    if img is None:
        print("Error: Could not load image")
        return 1
    
    # Display image details
    print(f"Image details: format={img.format}, size={img.size}, mode={img.mode}")
    
    # Preprocess the image
    img_array = preprocess_image(img)
    if img_array is None:
        print("Error: Could not preprocess image")
        return 1
    
    # Make prediction
    result = predict_captcha(model, img_array)
    if result is None:
        print("Error: Could not make prediction")
        return 1
    
    print(f"Captcha solution: {result}")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 