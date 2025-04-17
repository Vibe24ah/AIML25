from ultralytics import YOLO


# Load the YOLOv8n model (pretrained on COCO)
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' or 'yolov8m.pt' for larger models if needed

# Train the model on your custom dataset
model.train(
    data='/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/dataset.yaml',  # Path to the dataset YAML file
    epochs=3,                     # Number of training epochs
    imgsz=640,                     # Image size for training
    batch=16,                      # Batch size
    name='yolov8_nested',    # Name of the training run
    project='/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/runs/detect'  # Save directory
)

# Save the model after training 
model.export(
    format='pt',  # Save the model in PyTorch format
    weights='/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/runs/detect/yolov8_nested/weights/yolov8_nested.pt'
)


# Load the trained YOLOv8 model
#model = YOLO('yolov8_nested_image.pt')  # Path to the saved model

