import cv2
import numpy as np
import os
import glob
from tt_constants import BASE_RADS_PATH, RESIZED_GS_PATH, STANDARIZED_GS_PATH
from tt_utils import create_output_directory


def _calculate_global_stats(input_path):
    images = []
    # Cargar todas las imágenes del dataset
    for filename in glob.glob(os.path.join(input_path, '*.png')):  # Asegúrate de ajustar la extensión de archivo si es necesario
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img_float = img.astype(np.float32)
            images.append(img_float)
    
    # Concatenar todas las imágenes en un solo array para cálculo de estadísticas
    all_images = np.stack(images)
    
    # Calcular la media y la desviación estándar
    mean_global = np.mean(all_images)
    stdev_global = np.std(all_images)
    
    return mean_global, stdev_global


def get_mean_and_stdev(dataset):
    stats_path = f"{BASE_RADS_PATH}/{dataset}_stats.txt"

    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = f.read().split(',')
            mean_global = float(stats[0])
            stdev_global = float(stats[1])

    else:
        mean_global, stdev_global = _calculate_global_stats(RESIZED_GS_PATH)

        with open(stats_path, 'w') as file:
                file.write(f"{mean_global},{stdev_global}")

    
    return mean_global, stdev_global
    


def standardize_image(image_path,  mean_global, stdev_global):
    # Leer los valores de media y desviación estándar    
    # Leer y procesar la imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32)
    
    # Estandarizar la imagen
    standardized_img = (img_float - mean_global) / (stdev_global + 1e-6)
    
    # Reescalar los valores a 0-255 para visualización
    standardized_img_rescaled = cv2.normalize(standardized_img, None, 0, 255, cv2.NORM_MINMAX)
    standardized_img_uint8 = standardized_img_rescaled.astype(np.uint8)
    
    return standardized_img_uint8


dataset = "resized_gs"

create_output_directory(STANDARIZED_GS_PATH)

mean_global, stdev_global = get_mean_and_stdev(dataset)

for i, image_name in enumerate(sorted(os.listdir(RESIZED_GS_PATH))):
    if image_name.endswith(".png"):  # Assuming images are PNG format
        
        standarized_image = standardize_image(os.path.join(RESIZED_GS_PATH, image_name), mean_global, stdev_global)

        cv2.imwrite(f"{STANDARIZED_GS_PATH}/normalized_{image_name}", standarized_image)
