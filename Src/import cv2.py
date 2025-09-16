import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# Folder containing input images
image_folder = r"E:\Deepseamining(donotdelete)\Imagedeepsea"

# Folder to save segmented images
output_folder = r"E:\Deepseamining(donotdelete)\SegmentedImages"
os.makedirs(output_folder, exist_ok=True)

# Get all image files (jpg, jpeg, png)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Number of clusters for KMeans
k = 3

# Loop through each image
for file_name in image_files:
    print(f"Processing {file_name}...")

    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {file_name}, skipping.")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    # Save the original shape
    original_shape = image.shape

    # Flatten the image to 2D for clustering
    pixels = image.reshape((-1, 3))

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Replace each pixel with its cluster center
    new_colors = kmeans.cluster_centers_.astype("uint8")[kmeans.labels_]
    new_image = new_colors.reshape(original_shape)

    # Save the segmented image
    output_path = os.path.join(output_folder, f"segmented_{file_name}")
    cv2.imwrite(output_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

print("✅ All images processed and saved.")