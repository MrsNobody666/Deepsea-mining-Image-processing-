import cv2
import numpy as np
import os
import csv

# === CONFIGURATION ===
input_folder = "E:/Deepseamining(donotdelete)/SkeletonImages"
output_folder = "E:/Deepseamining(donotdelete)/Results"
os.makedirs(output_folder, exist_ok=True)

# Minimum area threshold to filter noise (in pixels)
MIN_AREA = 20

# === PROCESS EACH IMAGE IN FOLDER ===
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_name in image_files:
    print(f"\nProcessing: {image_name}")
    input_path = os.path.join(input_folder, image_name)
    output_image_path = os.path.join(output_folder, f"labeled_{image_name}")
    output_csv_path = os.path.join(output_folder, f"data_{os.path.splitext(image_name)[0]}.csv")

    # Load grayscale image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not read {input_path}")
        continue

    # Threshold to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # CSV writer
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Area", "Center X", "Center Y", "X", "Y", "Width", "Height"])

        # Process each component (skip background)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if area < MIN_AREA:
                continue  # Skip noise

            # Draw bounding box and ID
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"ID:{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Save info to CSV
            writer.writerow([i, area, int(cx), int(cy), x, y, w, h])

            # Console output
            print(f"✔ ID:{i} | Area: {area} px | Center: ({int(cx)}, {int(cy)}) | Box: ({x},{y},{w},{h})")

    # Save labeled image
    cv2.imwrite(output_image_path, output)
    print(f"✅ Saved labeled image → {output_image_path}")
    print(f"✅ Saved CSV → {output_csv_path}")

