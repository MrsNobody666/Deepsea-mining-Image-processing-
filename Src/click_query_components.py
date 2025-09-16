import cv2
import os
import csv

# === CONFIGURATION ===
results_folder = "E:/Deepseamining(donotdelete)/Results"
MIN_AREA = 20  # Ignore small components

# === STEP 1: Select a labeled image ===
image_files = [f for f in os.listdir(results_folder) if f.startswith("labeled_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("Available labeled images:")
for i, f in enumerate(image_files):
    print(f"{i + 1}. {f}")

choice = int(input(f"\nSelect an image (1-{len(image_files)}): ").strip()) - 1
if choice < 0 or choice >= len(image_files):
    print("❌ Invalid selection.")
    exit()

selected_img = image_files[choice]
base_name = os.path.splitext(selected_img.replace("labeled_", ""))[0]
csv_file = f"data_{base_name}.csv"

image_path = os.path.join(results_folder, selected_img)
csv_path = os.path.join(results_folder, csv_file)

# === STEP 2: Load image and component data ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

components = {}

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ID = int(row["ID"])
        area = int(row["Area"])
        if area < MIN_AREA:
            continue  # Skip small pebbles

        x, y, w, h = int(row["X"]), int(row["Y"]), int(row["Width"]), int(row["Height"])
        cx, cy = int(row["Center X"]), int(row["Center Y"])

        components[ID] = {
            "box": (x, y, w, h),
            "center": (cx, cy),
            "area": area
        }

# === STEP 3: Mouse click callback ===
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for ID, data in components.items():
            bx, by, bw, bh = data["box"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                cx, cy = data["center"]
                area = data["area"]

                print(f"\n🖱 You clicked on ID: {ID}")
                print(f"   Center     : ({cx}, {cy})")
                print(f"   Area       : {area} px")
                print(f"   Bounding Box: X={bx}, Y={by}, W={bw}, H={bh}")

                # Draw on a copy
                temp = image.copy()
                cv2.rectangle(temp, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.circle(temp, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(temp, f"ID:{ID}", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Interactive Viewer", temp)
                break

# === STEP 4: Show the window and set callback ===
cv2.namedWindow("Interactive Viewer", cv2.WINDOW_NORMAL)
cv2.imshow("Interactive Viewer", image)
cv2.setMouseCallback("Interactive Viewer", click_event)
print("\n🖱 Click on any box in the image window to display its info.")

cv2.waitKey(0)
cv2.destroyAllWindows()
