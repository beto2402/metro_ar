# import YOLO model
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)
model = YOLO('runs/classify/train4/weights/best.pt') # load best of training

# Train the model
model.train(data='iconos_100_2', epochs=5)