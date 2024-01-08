from ultralytics import YOLO

# Cargar modelo preentrenado y entrenar con las imágenes generados
YOLO('yolov8n-cls.pt').train(data='imagenes/da', epochs=5)

