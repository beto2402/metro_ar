import os
import cv2
import numpy as np
import random

import os
import cv2
import numpy as np
import random
from metro_predictor import MetroPredictor
from preprocess import get_canny


def add_gaussian_noise(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gauss
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(img, amount=0.04):
    row, col, _ = img.shape
    s_vs_p = 0.5
    out = np.copy(img)

    # Salt
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0

    return out

def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def random_brightness_transform(image, num_shapes=3, max_brightness_change=50):
    """
    Randomly changes the brightness of different areas in the image.
    
    :param image: The original image.
    :param num_shapes: Number of random shapes to use for changing brightness.
    :param max_brightness_change: Maximum change in brightness (positive or negative).
    :return: Transformed image.
    """
    transformed_image = image.copy()
    height, width, _ = transformed_image.shape

    for _ in range(num_shapes):
        # Randomly choose a rectangle or ellipse
        shape_type = random.choice(["rectangle", "ellipse"])
        
        # Randomly define the shape's dimensions and position
        x1, y1 = random.randint(0, width-10), random.randint(0, height-10)
        x2, y2 = random.randint(x1, width), random.randint(y1, height)
        
        # Random brightness change
        brightness_change = random.randint(-max_brightness_change, max_brightness_change)
        
        # Create a mask for the shape
        mask = np.zeros_like(transformed_image, dtype=np.uint8)
        if shape_type == "rectangle":
            cv2.rectangle(mask, (x1, y1), (x2, y2), (brightness_change, brightness_change, brightness_change), thickness=cv2.FILLED)
        else:  # ellipse
            cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (brightness_change, brightness_change, brightness_change), thickness=cv2.FILLED)
        
        # Apply the mask to the image
        transformed_image = cv2.add(transformed_image, mask)

    return transformed_image



import os
import cv2
import numpy as np
import random


def create_output_directory(base_folder, subfolder_name):
    output_dir = os.path.join(base_folder, subfolder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def add_gaussian_noise(image):
    row, col = image.shape
    mean = 0
    var = 500
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image


def transform_and_save(img, train_output_folder, test_output_folder, station_name, num_new_images=100, grayscale=False):
    current_image = img.copy()
    
    
    canny_image = get_canny(current_image)

    canny_image = add_gaussian_noise(canny_image)


    # We need to shuffle elements to ensure train and test data are representative
    possible_angles = random.sample(list(range(0, 359)), 180)
    train_elements_count = int(len(possible_angles) * 0.8)

    for i, angle in enumerate(possible_angles):
        # Decidir si la imagen va a la carpeta de entrenamiento o de prueba
        output_folder = train_output_folder if i < train_elements_count else test_output_folder

        filename = f"{station_name}.jpg"

        current_image = rotate_image(canny_image, angle)
        
        
        filename = f'rotate_{angle}_' + filename

        # Save the rotated image with transformations
        cv2.imwrite(os.path.join(output_folder, filename), current_image)


def get_transformed_img_path(base_output_folder, file_name):
    return os.path.join(base_output_folder, file_name)


# Tis is the function that needs to be called in order to generate the images
def augment_dataset(input_folder, output_base_folder):
    output_base_folder += '/canny'

    
    # Crear carpetas de train y test
    train_folder = create_output_directory(output_base_folder, 'train')
    test_folder = create_output_directory(output_base_folder, 'test')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
            img = cv2.imread(os.path.join(input_folder, filename))
            if img is None:
                continue

            
            station_name = filename.split('.')[0].replace(".jpg", "")
            

            train_output_folder = create_output_directory(train_folder, station_name)
            test_output_folder = create_output_directory(test_folder, station_name)

            transform_and_save(img, train_output_folder, test_output_folder, station_name)

