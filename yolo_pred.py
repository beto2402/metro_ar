# import YOLO model
from ultralytics import YOLO
import sys

# Load a model
model = YOLO('runs/classify/train5/weights/best.pt') # load a pretrained model (recommended for training)

# Ruta a la imagen que quieres predecir
image_path = 'iconos/2.1.png'

# Realizar la predicción
results = model.predict(image_path, show=True)

for result in results:
    breakpoint()
    for i, prob in enumerate(result.probs):
        print(i, prob)
