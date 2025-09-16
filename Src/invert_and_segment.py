import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# Input/output folders
image_folder = r"E:\Deepseamining(donotdelete)\Imagedeepsea"
output_folder = r"E:\Deepseamining(donotdelete)\SegmentedImages"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
k = 3  # number of clusters

for file_name in image_files:
    print(f"Processing {file_name}...")

    image_path = os.path.join(image_folder, file_name)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"Could not read {file_name}, skipping.")
        continue
s
    # ✅ Invert grayscale image
    inverted = cv2.bitwise_not(gray)

    # Convert to 3-channel for clustering (if needed)
    image_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)

    original_shape = image_rgb.shape
    pixels = image_rgb.reshape((-1, 3))

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_.astype("uint8")[kmeans.labels_]
    new_image = new_colors.reshape(original_shape)

    output_path = os.path.join(output_folder, f"segmented_{file_name}")
    cv2.imwrite(output_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

print("✅ All grayscale images inverted, clustered, and saved.")
