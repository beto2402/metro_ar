# import YOLO model
from ultralytics import YOLO
from preprocess import resize_and_save
import time
import asyncio
import cv2

model = YOLO(f"runs/classify/color/weights/best.pt")

def predecir(image_path):
    global model

    path_preprocesada = f"prediccion_{time.time()}.jpg"
    preprocesar(image_path, path_preprocesada)

    # Realizar la predicción
    result = model.predict(path_preprocesada)[0]

    top1_pred = result.probs.top1

    return result.names[top1_pred]


def preprocesar(og_path, new_path):
    img = cv2.imread(og_path, cv2.IMREAD_UNCHANGED)
        
    dim = (128, 128)
    
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite(new_path, resized_img)