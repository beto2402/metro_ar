import os
import cv2
import numpy as np


def crear_carpeta(carpeta, subcarpeta):
    carpeta_salida = os.path.join(carpeta, subcarpeta)

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    return carpeta_salida

def aumentar_datos(img, carpeta_entrenamiento, carpeta_prueba, id_estacion, n_imagenes):
    for i in range(n_imagenes):            
        imagen_modificada = img.copy()

        # Ruido Gaussiano
        mean = 0
        var = np.random.randint(100, 250)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, imagen_modificada.shape)
        noisy_img = imagen_modificada + gauss
        imagen_modificada = np.clip(noisy_img, 0, 255).astype(np.uint8)

        # Cambiar brillo a la imagen
        cambio = np.random.randint(-50, 51)
        hsv = cv2.cvtColor(imagen_modificada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, cambio)
        final_hsv = cv2.merge((h, s, v))
        imagen_modificada = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    

        # Pasar imagen a blanco y negro
        imagen_modificada = cv2.cvtColor(imagen_modificada, cv2.COLOR_BGR2GRAY)

        
        # Decidir si la imagen va a la carpeta de entrenamiento o de prueba
        carpeta_salida = carpeta_entrenamiento if i < int(n_imagenes * 0.8) else carpeta_prueba

        nuevo_nombre = f"{id_estacion}_{i}.jpg"

        # Guardar imagen en posicion normal
        cv2.imwrite(os.path.join(carpeta_salida, nuevo_nombre), imagen_modificada)

        # Rotar la imagen entre 1 y 359 grados
        centro_imagen = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(centro_imagen, np.random.choice(range(1, 359)), 1.0)
        imagen_modificada = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        nuevo_nombre = f'{np.random.choice(range(0, 359))}_' + nuevo_nombre

        # Save the rotated image with transformations
        cv2.imwrite(os.path.join(carpeta_salida, nuevo_nombre), imagen_modificada)



carpeta_entrada = 'imagenes/originales'
carpeta_salida = 'imagenes/da'


# Crear carpetas de entrenamiento y prueba
carpeta_entrenamiento = crear_carpeta(carpeta_salida, 'train')
carpeta_prueba = crear_carpeta(carpeta_salida, 'test')

for nombre_imagen in os.listdir(carpeta_entrada):

    img = cv2.imread(os.path.join(carpeta_entrada, nombre_imagen))
    
    id_estacion = nombre_imagen.replace(".jpg", "")
    
    carpeta_entrenamiento_estacion = crear_carpeta(carpeta_entrenamiento, id_estacion)
    carpeta_prueba_estacion = crear_carpeta(carpeta_prueba, id_estacion)

    aumentar_datos(img, carpeta_entrenamiento_estacion, carpeta_prueba_estacion, id_estacion, 150)

