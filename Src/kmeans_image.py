import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

output_folder = r"E:\Deepseamining(donotdelete)\SegmentedImages"
os.makedirs(output_folder, exist_ok=True)

# Folder containing images
image_folder = r"E:\Deepseamining(donotdelete)\Imagedeepsea"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Number of clusters
k = 5

# Loop through each image
for file_name in image_files:
    print(f"Processing {file_name}...")
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to 2D array of pixels
    original_shape = image.shape
    pixels = image.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Replace pixels with cluster centers
    new_colors = kmeans.cluster_centers_.astype("uint8")[kmeans.labels_]
    new_image = new_colors.reshape(original_shape)
    output_path = os.path.join(output_folder, f"segmented_{file_name}")
    cv2.imwrite(output_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"K-Means Segmented Image (k={k})")
    plt.imshow(new_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()