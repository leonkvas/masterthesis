import os
import shutil
from pathlib import Path
import cv2
import numpy as np

def get_captcha_type(image_path):
    """Determine captcha type based on saturation values."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    
    # Calculate average saturation
    avg_saturation = np.mean(saturation)
    
    # Determine type based on saturation thresholds
    if avg_saturation < 50:
        return "checkered"
    elif avg_saturation < 100:
        return "light_blue"
    else:
        return "dark_blue"

def organize_test2_images():
    # Define paths
    test2_dir = Path("data/test2")
    output_dir = Path("data/test2_organized")
    
    # Create output directories
    for captcha_type in ["checkered", "light_blue", "dark_blue"]:
        (output_dir / captcha_type).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in test2_dir.glob("*.jpg"):
        captcha_type = get_captcha_type(img_path)
        if captcha_type:
            dest_path = output_dir / captcha_type / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to {captcha_type} directory")

if __name__ == "__main__":
    organize_test2_images() 