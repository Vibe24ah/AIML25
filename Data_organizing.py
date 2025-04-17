import os
import cv2
import numpy as np
import shutil

# Paths
input_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Final dataset"
output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Labeled dataset"
os.makedirs(output_folder, exist_ok=True)

# Function to detect red boxes and generate YOLO labels
def detect_red_box_and_label(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the red box)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(largest_contour)

        # Convert to YOLO format
        x_center = (x + box_w / 2) / w
        y_center = (y + box_h / 2) / h
        norm_width = box_w / w
        norm_height = box_h / h

        # Save the label to a .txt file
        label_path = output_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")  # Class ID is 0

# Process all images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    # Copy the image to the output folder
    shutil.copy(input_path, output_path)

    # Detect red box and generate label
    detect_red_box_and_label(input_path, output_path)

print("✅ Labels generated successfully!")