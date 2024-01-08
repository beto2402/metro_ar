from ultralytics import YOLO
import time
import cv2
import os
from PIL import Image

model = YOLO("runs/classify/train/weights/best.pt")

path_preprocesadas = "imagenes\predicciones"

if not os.path.exists(path_preprocesadas):
    os.makedirs(path_preprocesadas)


def predecir(image_path):
    global model, path_preprocesadas

    path_img_preprocesada = f"{path_preprocesadas}/prediccion_{time.time()}.jpg"
    preprocesar(image_path, path_img_preprocesada)

    img = Image.open(path_img_preprocesada)

    # Realizar la predicción
    result = model.predict(img)[0]

    os.remove(image_path)

    top1_pred = result.probs.top1

    return result.names[top1_pred]


def preprocesar(og_path, nuevo_path):
    img = cv2.imread(og_path, cv2.IMREAD_UNCHANGED)
    
    img_redimensionada = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)

    img_blanco_negro = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2GRAY) 

    cv2.imwrite(nuevo_path, img_blanco_negro)

