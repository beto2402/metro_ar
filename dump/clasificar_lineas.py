from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Ruta al modelo guardado
model_path = 'modelo_lineas_metro.h5'

# Cargar el modelo
model = load_model(model_path)


def prepare_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convertir a un batch de tamaño 1
    img_array /= 255.0  # Normalizar si el modelo fue entrenado con imágenes normalizadas
    return img_array

# Preparar la imagen
image_path = 'iconos/2.18.png'

prepared_image = prepare_image(image_path, target_size=(133, 133))  # Asegúrate de que 'target_size' sea el mismo que se usó durante el entrenamiento

# Hacer la predicción
predictions = model.predict(prepared_image)

# Suponiendo una tarea de clasificación, obtener la clase con la mayor probabilidad
predicted_class = np.argmax(predictions, axis=1)

print("Clase predicha:", predicted_class)

