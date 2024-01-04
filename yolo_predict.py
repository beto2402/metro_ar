# import YOLO model
from ultralytics import YOLO
from preprocess import resize_and_save
import time
import asyncio


class YoloPredict:
    model = None

    def __init__(self):
        self.model = YOLO('runs/classify/train5/weights/best.pt')

    
    async def predict(self, image_path):
        preprocessed_img_path = f"predictions/prediction_{time.time()}.jpg"
        resize_and_save(image_path, preprocessed_img_path)


        # Realizar la predicción
        result = self.model.predict(image_path, show=True)[0]

        top1_pred = result.probs.top1

        return result.names[top1_pred]
    
