import os
import glob
import shutil

# Directories (adjust as needed)
labels_dir = 'data/valid'
images_dir = 'data/valid'
output_dir = 'labeled_captchas'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Class ID -> Character mapping
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


def parse_yolo_annotation(ann_file):
    """
    Reads a YOLO-format .txt file:
       class_id x_center y_center width height
    Returns a string label sorted by x_center (left-to-right).
    """
    boxes = []
    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                boxes.append((x_center, class_id))

    # Sort by x_center
    boxes_sorted = sorted(boxes, key=lambda x: x[0])

    # Convert each class_id to the corresponding character
    label_chars = [class_mapping.get(cls_id, '?') for (_, cls_id) in boxes_sorted]
    return ''.join(label_chars)


def sanitize_label(label):
    """
    Removes characters that might cause issues in filenames
    (e.g., slashes, colons, etc.).
    """
    return ''.join(ch for ch in label if ch.isalnum())


def main():
    # Collect all annotation files
    annotation_files = glob.glob(os.path.join(labels_dir, '*.txt'))

    # Possible image extensions
    image_exts = ['.jpg', '.png', '.jpeg']

    for ann_file in annotation_files:
        base_name = os.path.splitext(os.path.basename(ann_file))[0]
        # Derive the label
        captcha_label = parse_yolo_annotation(ann_file)
        label_clean = sanitize_label(captcha_label)  # remove invalid filename chars
        if label_clean == "":
            print(f"Warning: Empty label for annotation {ann_file}")
            continue

        # Find the corresponding image
        image_path = None
        for ext in image_exts:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"Warning: No image found for annotation {ann_file}")
            continue

        # Construct output filename
        # e.g. "AB12.png" if the label is "AB12"
        # If multiple images share the same label, you might add a counter
        out_name = label_clean + os.path.splitext(image_path)[1]  # keep extension
        out_path = os.path.join(output_dir, out_name)

        # Copy the image
        shutil.copy2(image_path, out_path)

        print(f"Copied {image_path} -> {out_path} (Label: {captcha_label})")


if __name__ == "__main__":
    main()
