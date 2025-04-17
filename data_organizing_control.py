import os
import cv2
import numpy as np
import shutil

# Paths
input_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Final dataset"
output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Labeled dataset"
check_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/check"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(check_folder, exist_ok=True)

# Function to detect red boxes and generate YOLO labels
def detect_red_box_and_label(image_path, output_path, check_path):
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

        # Get the minimum area rectangle for the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)  # Get the four corners of the rectangle
        box = np.int0(box)  # Convert to integer

        # Draw the yellow box on the image for checking
        check_image = image.copy()
        cv2.drawContours(check_image, [box], 0, (0, 255, 255), 2)  # Yellow box (BGR: 0, 255, 255)

        # Save the image with the yellow box to the check folder
        check_image_path = os.path.join(check_path, os.path.basename(image_path))
        cv2.imwrite(check_image_path, check_image)

        # Convert the rectangle to YOLO format
        x_center = rect[0][0] / w
        y_center = rect[0][1] / h
        norm_width = rect[1][0] / w
        norm_height = rect[1][1] / h

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

    # Detect red box, generate label, and create a check image
    detect_red_box_and_label(input_path, output_path, check_folder)

print("✅ Labels generated and check images created successfully!")