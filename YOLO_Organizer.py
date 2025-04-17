import os
import shutil
import random

# Paths
input_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Output_images"  # Replace with the actual path to your Output_images folder
output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Output_for_yolo"  # Replace with the desired path for Output_for_yolo
os.makedirs(output_folder, exist_ok=True)

# Create train/val folders
train_images = os.path.join(output_folder, "train/images")
train_labels = os.path.join(output_folder, "train/labels")
val_images = os.path.join(output_folder, "val/images")
val_labels = os.path.join(output_folder, "val/labels")

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Get all images and corresponding labels
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
label_files = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in image_files]

# Ensure only images with corresponding labels are included
data = [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(os.path.join(input_folder, lbl))]

# Split into train and validation sets (80% train, 20% val)
random.shuffle(data)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]

# Move files to train and val folders
for image_file, label_file in train_data:
    shutil.copy(os.path.join(input_folder, image_file), os.path.join(train_images, image_file))
    shutil.copy(os.path.join(input_folder, label_file), os.path.join(train_labels, label_file))

for image_file, label_file in val_data:
    shutil.copy(os.path.join(input_folder, image_file), os.path.join(val_images, image_file))
    shutil.copy(os.path.join(input_folder, label_file), os.path.join(val_labels, label_file))

print("✅ Dataset organized successfully into 'Output_for_yolo'!")