# import YOLO model
from ultralytics import YOLO
from preprocess import resize_and_save
import time
import asyncio


class YoloPredict:
    model = None
    grayscale = False

    @property
    def folder(self):
        if self.grayscale:
            return "g_scale_train_over_f"
        else:
            return "color"

    def __init__(self, grayscale=False):

        self.grayscale = grayscale
        self.model = YOLO(f"runs/classify/{self.folder}/weights/best.pt")
        


    async def predict(self, image_path):
        preprocessed_img_path = f"predictions/prediction_{time.time()}.jpg"
        resize_and_save(image_path, preprocessed_img_path, grayscale=self.grayscale)


        # Realizar la predicción
        result = self.model.predict(preprocessed_img_path)[0]

        top1_pred = result.probs.top1

        return result.names[top1_pred]
    
