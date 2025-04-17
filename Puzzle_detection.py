import cv2
import numpy as np
import os

def resize_and_crop_to_cover(img, target_w, target_h):
    h, w = img.shape[:2]
    img_aspect = w / h
    target_aspect = target_w / target_h

    # Resize while preserving aspect ratio
    if img_aspect > target_aspect:
        # Image is wider — scale height, then crop width
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h))
        x_offset = (new_w - target_w) // 2
        cropped = resized[:, x_offset:x_offset + target_w]
    else:
        # Image is taller — scale width, then crop height
        scale = target_w / w
        new_h = int(h * scale)
        resized = cv2.resize(img, (target_w, new_h))
        y_offset = (new_h - target_h) // 2
        cropped = resized[y_offset:y_offset + target_h, :]

    return cropped

# Coordinates of the four corners of the image area on the puzzle box
# Order: top-left, top-right, bottom-right, bottom-left
dst_points = np.array([
    [926, 1070],
    [1993, 1076],
    [1970, 2660],
    [929, 2644]
], dtype="float32")

# Compute width and height of the replacement image based on distances
width = int(max(
    np.linalg.norm(dst_points[0] - dst_points[1]),
    np.linalg.norm(dst_points[2] - dst_points[3])
))
height = int(max(
    np.linalg.norm(dst_points[0] - dst_points[3]),
    np.linalg.norm(dst_points[1] - dst_points[2])
))

# Source points (corners of the replacement image)
src_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Load the template puzzle box image
template = cv2.imread("IMG_9567.jpg")

# Folders
replacement_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Fred_pictures"
output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Output_images"


for idx, filename in enumerate(os.listdir(replacement_folder)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    replacement_img_path = os.path.join(replacement_folder, filename)
    replacement_image = cv2.imread(replacement_img_path)

    # Resize replacement to match target area
    # resized = cv2.resize(replacement_image, (width, height))
    # Resize with aspect ratio preserved (object-fit: cover style)
    resized = resize_and_crop_to_cover(replacement_image, width, height)

    replacement_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Fred_pictures"
    output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Output_images"

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the replacement image
    warped = cv2.warpPerspective(resized, matrix, (template.shape[1], template.shape[0]))

    # Create a mask from the warped image to blend properly
    mask = np.zeros_like(template, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(int), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # Blend the warped image into the original template
    background = cv2.bitwise_and(template, mask_inv)
    combined = cv2.add(background, cv2.bitwise_and(warped, mask))

    # Save final output
    output_path = os.path.join(output_folder, f"puzzle_{idx+1}.jpg")
    cv2.imwrite(output_path, combined)

print("Puzzle box images generated with precise perspective alignment!")