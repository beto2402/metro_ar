import os
import cv2
import numpy as np
import random

import os
import cv2
import numpy as np
import random

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

import os
import cv2
import numpy as np
import random


def create_output_directory(base_folder, subfolder_name):
    output_dir = os.path.join(base_folder, subfolder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def apply_random_transformations_and_save(img, train_output_folder, test_output_folder, base_filename, num_new_images=100):
    for i in range(num_new_images):
        choice = np.random.choice(['rotate', 'gaussian', 'saltpepper', 'brightness'])
        
        if choice == 'rotate':
            angle = np.random.choice([90, 180, 270])
            transformed_img = rotate_image(img, angle)
            suffix = f'_rotate_{angle}'
        elif choice == 'gaussian':
            transformed_img = add_gaussian_noise(img)
            suffix = '_gaussian'
        elif choice == 'saltpepper':
            transformed_img = add_salt_and_pepper_noise(img)
            suffix = '_saltpepper'
        elif choice == 'brightness':
            value = np.random.randint(-50, 51)
            transformed_img = change_brightness(img, value)
            suffix = '_brightness'
        
        # Decidir si la imagen va a la carpeta de entrenamiento o de prueba
        if i < int(num_new_images * 0.8):
            output_path = os.path.join(train_output_folder, f"{base_filename}{suffix}_{i}.jpg")
        else:
            output_path = os.path.join(test_output_folder, f"{base_filename}{suffix}_{i}.jpg")

        cv2.imwrite(output_path, transformed_img)


input_folder = 'iconos'
output_base_folder = 'iconos_100_2'

# Crear carpetas de train y test
train_folder = create_output_directory(output_base_folder, 'train')
test_folder = create_output_directory(output_base_folder, 'test')

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        decimal_number = filename.split('_')[0].replace(".png", "")
        base_filename = filename.split('_')[0].replace(".png", "")

        train_output_folder = create_output_directory(train_folder, decimal_number)
        test_output_folder = create_output_directory(test_folder, decimal_number)

        apply_random_transformations_and_save(img, train_output_folder, test_output_folder, base_filename)
