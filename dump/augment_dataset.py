from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np
import cv2
import random

def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

# Directory where the original images are stored
original_images_dir = 'datasets/mios/resized'

# Directory where the augmented images will be saved
augmented_images_dir = 'dataset/mios/augs'

# Create an instance of ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each original image
for filename in os.listdir(original_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        img = load_img(os.path.join(original_images_dir, filename))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Image augmentation
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=augmented_images_dir,
                                  save_prefix='aug',
                                  save_format='jpg'):
            augmented_image = batch[0].astype('uint8')

            # Apply brightness adjustment
            if random.choice([True, False]):
                augmented_image = adjust_brightness(augmented_image, value=random.randint(-30, 30))

            # Apply noise
            if random.choice([True, False]):
                augmented_image = add_noise(augmented_image)

            file_root, file_ext = os.path.splitext(filename)
            new_filename = f"{file_root}__aug_{i}{file_ext}"

            cv2.imwrite(os.path.join(augmented_images_dir, new_filename), augmented_image)
            
            i += 1
            if i >= 200:  # Generate 200 images per original image
                break
