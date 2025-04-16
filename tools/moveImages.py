import os
import random
import shutil
import time
# Source and destination directories
source_dir = "data/val2"
destination_dir = "data/train2"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all image files in the source directory
image_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(len(image_files))
#print("TEST")
#print(2604*0.8, len(image_files) * 0.8)
#time.sleep(1000)
total = 2412

trainCount = total * 0.8
testCount = (total - trainCount) * 0.5
valCount = total - trainCount - testCount

print(total, trainCount, testCount, valCount)

# Randomly select split
images_to_move = random.sample(image_files, int(trainCount))

# Move the selected images to the destination directory
for img_file in images_to_move:
    source_path = os.path.join(source_dir, img_file)
    destination_path = os.path.join(destination_dir, img_file)
    shutil.move(source_path, destination_path)

print(f"Moved {len(images_to_move)} images to {destination_dir}.")
