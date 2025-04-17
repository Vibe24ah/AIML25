from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/runs/detect/yolov8_nested/weights/best.pt')  # Path to the trained model weights

# Path to the test image
test_image_path = '/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Test_images/IMG_9568.jpg'

# Run inference
results = model.predict(source=test_image_path, save=True, save_txt=True)

# Load the test image
image = cv2.imread(test_image_path)

# Plot the detected bounding box
for result in results:
    for box in result.boxes.xyxy:  # Bounding box coordinates in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = map(int, box.tolist())
        print(f"Bounding Box Coordinates: Top-Left ({x1}, {y1}), Bottom-Right ({x2}, {y2})")

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for detected bounding box
        # Optionally, add text for the class ID
        cv2.putText(image, "nested_image", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with the detected bounding box
output_image_path = '/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Test_images/IMG_9567_with_box.jpg'
cv2.imwrite(output_image_path, image)

print(f"✅ Detection complete! Image with bounding box saved at: {output_image_path}")