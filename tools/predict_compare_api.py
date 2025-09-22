import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from inference_sdk import InferenceHTTPClient

# --- Parameters ---
IMG_SIZE = (50, 250)
BATCH_SIZE = 32

# Define the vocabulary: digits 0-9 and uppercase A-Z
vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
vocab_size = len(vocab) + 1  # +1 for padding (0)
char_to_idx = {char: i + 1 for i, char in enumerate(vocab)}  # Mapping: '0'->1, etc.
idx_to_char = {i + 1: char for i, char in enumerate(vocab)}
max_captcha_len = 7

# API configuration
ROBOFLOW_API_KEY = "" # needs to be filled
MODEL_ID = "text-captchas/2"

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)


def load_image(file_path):
    """
    Loads and preprocesses image for model prediction.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1)  # Using grayscale (1 channel)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_with_model(model, img):
    """
    Makes a prediction using the trained model.
    """
    predictions = model.predict(img)
    pred_indices = tf.argmax(predictions, axis=-1).numpy()[0]
    
    # Convert indices to characters (skip padding, which is 0)
    pred_chars = [idx_to_char.get(idx, '') for idx in pred_indices if idx != 0]
    predicted_text = ''.join(pred_chars)
    return predicted_text


def predict_with_api(file_path):
    """
    Makes a prediction using the Roboflow API via the inference SDK.
    """
    try:
        #print(f"Sending image {os.path.basename(file_path)} to Roboflow API...")
        # Make the API call using the SDK
        result = CLIENT.infer(file_path, model_id=MODEL_ID)
        
        # Debug - print full response
        #print(f"API response: {result}")
        
        # Extract predictions
        if 'predictions' in result and result['predictions']:
            # Sort predictions by x coordinate (left to right)
            sorted_predictions = sorted(result['predictions'], key=lambda p: p['x'])
            
            # Extract class names and clean up special cases
            predictions = []
            for pred in sorted_predictions:
                if 'class' in pred:
                    # Clean up the class name (handle "0 zero" special case)
                    class_name = pred['class']
                    if class_name == "0 zero":
                        class_name = "0"
                    # Take just the first character for any other multi-character class
                    elif len(class_name) > 1:
                        class_name = class_name[0]
                    
                    predictions.append(class_name)
            
            # Join all detected text pieces
            predicted_text = ''.join(predictions)
            if not predicted_text:
                return "No text detected in API predictions"
            return predicted_text
        else:
            print(f"API response had no predictions: {result}")
            return "No text detected by API"
    
    except Exception as e:
        print(f"Exception during API call: {str(e)}")
        return f"API Error: {str(e)}"


def extract_ground_truth(file_path):
    """
    Extracts ground truth label from the file name.
    """
    file_name = os.path.basename(file_path)
    ground_truth = os.path.splitext(file_name)[0]
    return ground_truth


def evaluate_predictions(test_dir, model_path, use_api=True):
    """
    Evaluates model and API predictions on test dataset.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Compile the model with custom metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get all test images
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.endswith(('.jpg', '.png'))]
    
    results = []
    
    for img_path in test_images:
        ground_truth = extract_ground_truth(img_path)
        
        # Model prediction
        img = load_image(img_path)
        model_prediction = predict_with_model(model, img)
        model_correct = model_prediction == ground_truth
        
        result = {
            'image': os.path.basename(img_path),
            'ground_truth': ground_truth,
            'model_prediction': model_prediction,
            'model_correct': model_correct
        }
        
        # API prediction if enabled
        if use_api:
            api_prediction = predict_with_api(img_path)
            api_correct = api_prediction == ground_truth
            result.update({
                'api_prediction': api_prediction,
                'api_correct': api_correct
            })
        
        results.append(result)
        
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Model Prediction: {model_prediction} ({'Correct' if model_correct else 'Incorrect'})")
        if use_api:
            print(f"API Prediction: {api_prediction} ({'Correct' if api_correct else 'Incorrect'})")
        print("----------------------------")
    
    # Calculate accuracy
    model_accuracy = sum(r['model_correct'] for r in results) / len(results)
    
    print(f"\nResults Summary:")
    print(f"Total images tested: {len(results)}")
    print(f"Model accuracy: {model_accuracy:.2%}")
    
    if use_api:
        api_accuracy = sum(r['api_correct'] for r in results) / len(results)
        print(f"API accuracy: {api_accuracy:.2%}")
    
    return results


def visualize_results(results):
    """
    Visualizes comparison between model and API predictions.
    """
    # Count how many times each system was correct
    model_only = sum(r['model_correct'] and not r['api_correct'] for r in results)
    api_only = sum(not r['model_correct'] and r['api_correct'] for r in results)
    both_correct = sum(r['model_correct'] and r['api_correct'] for r in results)
    both_incorrect = sum(not r['model_correct'] and not r['api_correct'] for r in results)
    
    # Create a bar chart
    categories = ['Model only', 'API only', 'Both correct', 'Both incorrect']
    values = [model_only, api_only, both_correct, both_incorrect]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Comparison of Model vs API Predictions')
    plt.ylabel('Number of examples')
    plt.savefig('prediction_comparison.png')
    plt.show()


def test_single_image_api(image_path):
    """
    Test the API with a single image and print detailed debug information.
    """
    print(f"\n--- Testing API with single image: {image_path} ---")
    result = predict_with_api(image_path)
    print(f"Final API result: {result}")
    return result


if __name__ == "__main__":
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "..", "data", "test")
    model_path = os.path.join(base_dir, "../", "captcha_cnn_model.keras")
    
    # Flag to control whether API calls are made
    USE_API = True  # Set to False to skip API calls
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please provide the correct path to the trained model.")
        exit(1)
    
    # First test with a single image to debug the API call
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                   if f.endswith(('.jpg', '.png'))]
    
    if test_images:
        print("\n=== TESTING WITH A SINGLE IMAGE FIRST ===")
        first_image = test_images[0]
        test_single_image_api(first_image)
        
        print("\nDo you want to continue with all images? (y/n)")
        response = input().lower()
        if response != 'y':
            print("Exiting as requested.")
            exit(0)
    
    # Evaluate on all test images
    print(f"\n=== EVALUATING ON ALL TEST IMAGES ===")
    print(f"Evaluating model{' and API' if USE_API else ''} on test images in: {test_dir}")
    
    results = evaluate_predictions(test_dir, model_path, USE_API)
    
    # Visualize results if API was used
    if USE_API:
        visualize_results(results)
    
    # Save results to JSON file
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to 'prediction_results.json'") 