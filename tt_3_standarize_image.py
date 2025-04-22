import cv2
import numpy as np
import os
from tt_constants import RESIZED_GS_PATH, NORMALIZED_GS_PATH
from tt_utils import create_output_directory


def normalize_images(images):
    normalized_images = []
    for img in images:
        # Find minimum and maximum pixel values
        min_val = np.min(img)
        max_val = np.max(img)
        # Normalize pixel values
        normalized_img = (img - min_val) / (max_val - min_val)
        normalized_images.append(normalized_img)
    return normalized_images

# Load dataset
images = []
for filename in sorted(os.listdir(RESIZED_GS_PATH)):
    if filename.endswith(".png"):  # Assuming images are PNG format
        img = cv2.imread(os.path.join(RESIZED_GS_PATH, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

# Normalize images
normalized_images = normalize_images(images)

create_output_directory(NORMALIZED_GS_PATH)

# Save or use normalized images
# For example, you can save them back to disk:
for i, img in enumerate(normalized_images):
    cv2.imwrite(f"{NORMALIZED_GS_PATH}/normalized_image_{i}.png", img * 255)  # Convert back to 0-255 range for saving as PNG
