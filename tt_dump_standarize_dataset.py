import cv2
import numpy as np
import glob
from tt_constants import RESIZED_GS_PATH, BASE_RADS_PATH
import os


def format_mean_and_stdev(mean, stdev):
    return {
        "mean": mean,
        "stdev": stdev
    }


def calculate_dataset_mean_and_stdev():
    # Lista para almacenar todas las imágenes como arrays
    images = []

    # Cargar todas las imágenes del dataset
    for filename in glob.glob(f'{RESIZED_GS_PATH}/*.png'):  # Asegúrate de que la ruta y la extensión sean correctas
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_float = img.astype(np.float32)
        images.append(img_float)

    # Concatenar todas las imágenes en un solo array para cálculo de estadísticas
    all_images = np.stack(images)

    # Calcular la media y la desviación estándar
    return format_mean_and_stdev(np.mean(all_images), np.std(all_images))



def get_dataset_mean_and_stdev():
    dataset_mean_and_stdev = {}

    if os.path.exists(f"{BASE_RADS_PATH}/mean") and os.path.exists(f"{BASE_RADS_PATH}/stdev"):
        # Try to open and read the file
        mean = float(open(f"{BASE_RADS_PATH}/mean", 'r').read())  # Reading the value as a number
        stdev = float(open(f"{BASE_RADS_PATH}/stdev", 'r').read())  # Reading the value as a number

        return format_mean_and_stdev(mean, stdev)

    else:
        dataset_mean_and_stdev = calculate_dataset_mean_and_stdev()

        for key in dataset_mean_and_stdev:
            # If the file is not found, create the file and execute some code
            with open(f"{BASE_RADS_PATH}/{key}", 'w') as file:
                file.write(str(dataset_mean_and_stdev[key]))
                
                print(f"{key} not found, created a new file with initial value: {dataset_mean_and_stdev[key]}")
        
        return dataset_mean_and_stdev

get_dataset_mean_and_stdev()