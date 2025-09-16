import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
from skimage.util import invert as sk_invert

# Input/output folders
image_folder = r"E:\Deepseamining(donotdelete)\SegmentedImages"
output_folder = r"E:\Deepseamining(donotdelete)\SkeletonImages"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for file_name in image_files:
    print(f"Processing {file_name}...")

    image_path = os.path.join(image_folder, file_name)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"Could not read {file_name}, skipping.")
        continue

    # Threshold to binary (important for skeletonize)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Convert to boolean: skeletonize expects True for foreground
    binary_bool = binary > 0

    # Skeletonize
    skeleton = skeletonize(binary_bool)

    # Convert back to uint8 for saving
    skeleton_img = (skeleton * 255).astype(np.uint8)

    output_path = os.path.join(output_folder, f"skeleton_{file_name}")
    cv2.imwrite(output_path, skeleton_img)

print("✅ All inverted images converted to skeletons and saved.")


