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
    out = np.copy(img)
    # Salt
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0
    return out

def add_speckle_noise(img):
    gauss = np.random.randn(*img.shape)
    noisy_img = img + img * gauss
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_poisson_noise(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_localvar_noise(img):
    localvar = img.astype(np.float64) / 100.0
    noisy_img = np.random.normal(loc=img, scale=np.sqrt(localvar))
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_random_noise(img):
    noise = np.random.random(img.shape) * 255
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_cosine_noise(img):
    row, col, ch = img.shape
    cos_noise = 20 * (np.cos(np.arange(row) / row * 2 * np.pi) - 1)
    cos_noise = np.repeat(cos_noise[:, np.newaxis], col, axis=1)
    cos_noise = np.repeat(cos_noise[:, :, np.newaxis], ch, axis=2)
    noisy_img = img + cos_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_multiplicative_noise(img):
    noise = np.random.randn(*img.shape)
    noisy_img = img * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_impulse_noise(img):
    row, col, ch = img.shape
    number_of_pixels = random.randint(300, 10000)
    for _ in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    return img

def add_rayleigh_noise(img):
    mode = 50
    row, col, ch = img.shape
    rayleigh = np.random.rayleigh(mode, img.shape)
    noisy_img = img + rayleigh
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

import os
import cv2
import numpy as np

def add_gaussian_noise(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gauss
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(img, amount=0.04):
    out = np.copy(img)
    # Salt
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0
    return out

# Otras funciones de ruido como add_speckle_noise, add_poisson_noise, etc.

def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def apply_random_transformations(img, num_new_images=20):
    transformed_images = []
    for _ in range(num_new_images):
        choice = np.random.choice(['rotate', 'gaussian', 'saltpepper', 'brightness'])
        
        if choice == 'rotate':
            angle = np.random.choice([90, 180, 270])
            transformed_img = rotate_image(img, angle)
        elif choice == 'gaussian':
            transformed_img = add_gaussian_noise(img)
        elif choice == 'saltpepper':
            transformed_img = add_salt_and_pepper_noise(img)
        elif choice == 'brightness':
            value = np.random.randint(-50, 51)
            transformed_img = change_brightness(img, value)
        
        transformed_images.append(transformed_img)

    return transformed_images

input_folder = 'iconos'
output_folder = 'iconos_ruido'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        base_filename = os.path.splitext(filename)[0]

        new_images = apply_random_transformations(img)
        for idx, new_img in enumerate(new_images):
            cv2.imwrite(os.path.join(output_folder, f"{base_filename}_transformed_{idx}.jpg"), new_img)


for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        base_filename = os.path.splitext(filename)[0]

        # Aplicar diferentes tipos de ruido y guardar
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_gaussian.jpg"), add_gaussian_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_saltpepper.jpg"), add_salt_and_pepper_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_speckle.jpg"), add_speckle_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_poisson.jpg"), add_poisson_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_localvar.jpg"), add_localvar_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_random.jpg"), add_random_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_cosine.jpg"), add_cosine_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_multiplicative.jpg"), add_multiplicative_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_impulse.jpg"), add_impulse_noise(img))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_rayleigh.jpg"), add_rayleigh_noise(img))

