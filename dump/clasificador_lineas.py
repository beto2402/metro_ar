import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images_and_labels(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (133, 133))  # Asegúrate de que todas las imágenes tengan el mismo tamaño
                images.append(img)
                label = int(filename.split('_')[0].split('.')[0])  # Extraer la parte entera para la línea del metro
                labels.append(label)
    images = np.array(images) / 255.0  # Normalizar las imágenes
    labels = np.array(labels)
    return images, labels

input_folder = 'iconos_ruido'
images, labels = load_images_and_labels(input_folder)

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(133, 133, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')  # Asumiendo 12 líneas de metro
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2],  # Ajuste en el brillo
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

y_train_adjusted = y_train - 1
y_val_adjusted = y_val - 1

# Entrenar el modelo
history = model.fit(X_train, y_train_adjusted,
                    epochs=10,
                    validation_data=(X_val, y_val_adjusted))

# Guardar el modelo entrenado
model.save("modelo_lineas_metro_color.h5")  # Guarda el modelo como un archivo H5
