import cv2
import os
import csv

# === CONFIG ===
results_folder = "E:/Deepseamining(donotdelete)/Results"
image_exts = (".png", ".jpg", ".jpeg")

# === STEP 1: Get all labeled images ===
image_files = [f for f in os.listdir(results_folder) if f.lower().startswith("labeled_") and f.lower().endswith(image_exts)]

if not image_files:
    print("❌ No labeled images found in Results folder.")
    exit()

# === STEP 2: Let user pick an image ===
print("🖼 Available labeled images:")
for idx, file in enumerate(image_files):
    print(f"  {idx + 1}. {file}")

img_index = input(f"\nSelect an image (1-{len(image_files)}): ").strip()
if not img_index.isdigit() or not (1 <= int(img_index) <= len(image_files)):
    print("❌ Invalid selection.")
    exit()

selected_image = image_files[int(img_index) - 1]
base_name = os.path.splitext(selected_image.replace("labeled_", ""))[0]
csv_name = f"data_{base_name}.csv"

image_path = os.path.join(results_folder, selected_image)
csv_path = os.path.join(results_folder, csv_name)

# === STEP 3: Load the image ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# === STEP 4: Load CSV data and map short IDs ===
actual_to_display = {}     # Maps actual ID -> short ID
display_to_actual = {}     # Maps short ID -> actual ID
component_data = {}

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        actual_id = int(row["ID"])
        display_id = idx  # 1, 2, 3...
        actual_to_display[actual_id] = display_id
        display_to_actual[display_id] = actual_id
        component_data[actual_id] = {
            "area": int(row["Area"]),
            "center": (int(row["Center X"]), int(row["Center Y"])),
            "box": (int(row["X"]), int(row["Y"]), int(row["Width"]), int(row["Height"]))
        }

# === STEP 5: Show original image with short IDs ===
image_with_short_ids = image.copy()
for actual_id, data in component_data.items():
    display_id = actual_to_display[actual_id]
    x, y, w, h = data["box"]
    cv2.rectangle(image_with_short_ids, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(image_with_short_ids, f"{display_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow("Labeled Image with Short IDs", image_with_short_ids)
cv2.waitKey(1)

# === STEP 6: Query Loop (use short IDs) ===
print(f"\n🔎 Now querying components from: {selected_image}")
while True:
    user_input = input("Enter a short display ID (or 'q' to quit): ").strip()
    
    if not user_input:
        print("❌ Please enter a valid ID or 'q' to quit.")
        continue

    if user_input.lower() == 'q':
        break

    if not user_input.isdigit():
        print("❌ Please enter a numeric ID.")
        continue

    display_id = int(user_input)
    if display_id not in display_to_actual:
        print(f"❌ Display ID {display_id} not found.")
        continue

    actual_id = display_to_actual[display_id]
    data = component_data[actual_id]
    cx, cy = data["center"]
    x, y, w, h = data["box"]

    print(f"\n✔ Component ID {display_id} (Actual ID: {actual_id})")
    print(f"   Center      : ({cx}, {cy})")
    print(f"   Area        : {data['area']} px")
    print(f"   Bounding Box: X={x}, Y={y}, Width={w}, Height={h}")

    # === Highlight selected component ===
    highlight_img = image.copy()
    cv2.rectangle(highlight_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(highlight_img, f"ID:{display_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.circle(highlight_img, (cx, cy), 4, (255, 0, 0), -1)

    # === Show and save highlighted image ===
    cv2.imshow("Selected Component", highlight_img)
    highlight_save_path = os.path.join(results_folder, f"highlight_DisplayID_{display_id}_{base_name}.png")
    cv2.imwrite(highlight_save_path, highlight_img)
    print(f"💾 Saved to: {highlight_save_path}")

cv2.destroyAllWindows()
1