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

cv2.imshow("Labeled Image", image)
cv2.waitKey(1)  # non-blocking display

# === STEP 4: Load CSV data ===
component_data = {}
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ID = int(row["ID"])
        component_data[ID] = {
            "area": int(row["Area"]),
            "center": (int(row["Center X"]), int(row["Center Y"])),
            "box": (int(row["X"]), int(row["Y"]), int(row["Width"]), int(row["Height"]))
        }

# === STEP 5: Query Loop ===
print(f"\n🔎 Now querying components from: {selected_image}")
while True:
    user_input = input("Enter a component ID (or 'q' to quit): ").strip()
    
    if user_input.lower() == 'q':
        break

    if not user_input.isdigit():
        print("❌ Please enter a valid numeric ID.")
        continue

    query_id = int(user_input)
    if query_id not in component_data:
        print(f"❌ Component ID {query_id} not found.")
    else:
        data = component_data[query_id]
        cx, cy = data["center"]
        x, y, w, h = data["box"]
        print(f"\n✔ Component {query_id}")
        print(f"   Center     : ({cx}, {cy})")
        print(f"   Area       : {data['area']} px")
        print(f"   Bounding Box: X={x}, Y={y}, Width={w}, Height={h}")

cv2.destroyAllWindows()
