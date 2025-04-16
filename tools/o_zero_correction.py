import os
import sys
import tensorflow as tf
import numpy as np
from inference_sdk import InferenceHTTPClient
import shutil

# --- API configuration ---
ROBOFLOW_API_KEY = "TglwaRju23eGCaereXhC"
MODEL_ID = "text-captchas/2"

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

def get_o_zero_images(data_dir):
    """
    Find all images in the dataset that have O or 0 in their filename.
    """
    o_zero_images = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            if '0' in base_name or 'O' in base_name:
                o_zero_images.append(os.path.join(data_dir, filename))
    
    return o_zero_images

def predict_with_api(file_path):
    """
    Makes a prediction using the Roboflow API via the inference SDK.
    """
    try:
        # Make the API call using the SDK
        result = CLIENT.infer(file_path, model_id=MODEL_ID)
        
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
                return None
            return predicted_text
        else:
            return None
    
    except Exception as e:
        print(f"Exception during API call for {file_path}: {str(e)}")
        return None

def analyze_and_correct_labels(image_paths):
    """
    Analyze each image with O or 0 in the filename and suggest corrections.
    """
    corrections = {}
    skipped = []
    
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(base_name)[0]
        print(f"Processing: {base_name}")
        
        # Get API prediction
        api_prediction = predict_with_api(image_path)
        
        if api_prediction is None:
            print(f"  - Failed to get API prediction, skipping")
            skipped.append(image_path)
            continue
        
        # Check for O/0 mismatches
        new_label = list(filename_without_ext)
        made_correction = False
        
        # Handle length mismatch cases
        if len(api_prediction) != len(filename_without_ext):
            print(f"  - Length mismatch: filename={filename_without_ext}, prediction={api_prediction}")
            
            # Try to find matching positions for O and 0 characters
            possible_correction = False
            for i, char in enumerate(filename_without_ext):
                if char in ['O', '0']:
                    # Look for corresponding O/0 in the prediction
                    # Strategy: check characters at same position and nearby positions
                    possible_positions = []
                    
                    # Check exact position if it's within bounds
                    if i < len(api_prediction):
                        possible_positions.append(i)
                    
                    # Check adjacent positions too if within bounds
                    if i > 0 and i-1 < len(api_prediction):
                        possible_positions.append(i-1)
                    if i+1 < len(api_prediction):
                        possible_positions.append(i+1)
                    
                    # Check if any of these positions has an O or 0
                    for pos in possible_positions:
                        pred_char = api_prediction[pos]
                        if pred_char in ['O', '0'] and pred_char != char:
                            possible_correction = True
                            break
                    
                    if possible_correction:
                        break
            
            if possible_correction:
                print(f"  - Possible O/0 correction found but length mismatch, skipping for safety")
                skipped.append(image_path)
                continue
            else:
                print(f"  - No O/0 correction needed despite length mismatch")
        else:
            # For matching lengths, check character by character
            for i in range(len(filename_without_ext)):
                # Only correct O and 0
                if filename_without_ext[i] in ['O', '0'] and api_prediction[i] in ['O', '0']:
                    if filename_without_ext[i] != api_prediction[i]:
                        old_char = filename_without_ext[i]
                        new_char = api_prediction[i]
                        new_label[i] = new_char
                        made_correction = True
                        print(f"  - Char at position {i+1}: '{old_char}' should be '{new_char}'")
        
        if made_correction:
            new_label_str = ''.join(new_label)
            corrections[image_path] = new_label_str
            print(f"  - Suggested correction: {filename_without_ext} -> {new_label_str}")
        else:
            print(f"  - No O/0 corrections needed")
    
    return corrections, skipped

def apply_corrections(corrections, output_dir=None):
    """
    Save copies of the corrected images to a new directory.
    """
    if not corrections:
        print("No corrections to apply.")
        return
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    for image_path, new_label in corrections.items():
        file_ext = os.path.splitext(os.path.basename(image_path))[1]
        new_filename = new_label + file_ext
        new_path = os.path.join(output_dir, new_filename)
        
        # Copy the file with the new name
        shutil.copy2(image_path, new_path)
        print(f"Saved: {image_path} -> {new_path}")

if __name__ == "__main__":
    # Set fixed paths
    data_dir = "data/train"
    output_dir = "data/val2"
    
    print(f"Scanning directory: {data_dir}")
    o_zero_images = get_o_zero_images(data_dir)
    print(f"Found {len(o_zero_images)} images with O or 0 in their filenames")
    
    if not o_zero_images:
        print("No images to process. Exiting.")
        sys.exit(0)
    
    print("\nAnalyzing images and checking for O/0 misclassifications...")
    corrections, skipped = analyze_and_correct_labels(o_zero_images)
    
    print("\nAnalysis complete:")
    print(f"  - Found {len(corrections)} images with potential O/0 misclassifications")
    print(f"  - Skipped {len(skipped)} images due to prediction issues")
    
    print("\nSaving corrected images...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process corrected O/0 images
    apply_corrections(corrections, output_dir)
    
    # Create a set of both corrected and skipped paths to exclude from copying
    excluded_paths = set(corrections.keys()) | set(skipped)
    print(f"Total excluded paths: {len(excluded_paths)} (corrected + skipped)")
    
    # Copy all other images that were not corrected or skipped
    all_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    copied_count = 0
    for img_path in all_images:
        if img_path not in excluded_paths:  # Exclude both corrected and skipped images
            # Copy with the original filename
            filename = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, filename)
            
            # Only copy if the file doesn't already exist
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                copied_count += 1
    
    print(f"\nCorrected images saved to {output_dir}")
    print(f"Additionally copied {copied_count} uncorrected images (excluding skipped ones)")
    print(f"Total images in new dataset: {len(corrections) + copied_count}")
    print(f"  - {len(corrections)} images with O/0 corrections")
    print(f"  - {copied_count} other images (no skipped images)")
    print(f"  - {len(skipped)} skipped images were excluded") 