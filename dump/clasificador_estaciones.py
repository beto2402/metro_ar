import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, GlobalAveragePooling2D, Dense

def conv_block(x, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def residual_block(x, filters, repeats, use_projection=False):
    shortcut = x
    for i in range(repeats):
        if i == 0 and use_projection:
            shortcut = Conv2D(filters, 1, strides=2, padding='same')(x)
            shortcut = BatchNormalization()(shortcut)

        x = Conv2D(filters, 3, strides=2 if i == 0 else 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)

        if i == 0:
            x = Add()([x, shortcut])
        x = ReLU()(x)

    return x

def build_model(num_classes):
    inputs = Input(shape=(133, 133, 3))
    
    x = conv_block(inputs, 16, 3, 2)
    x = conv_block(x, 32, 3, 2)
    x = residual_block(x, 32, 1, True)
    x = conv_block(x, 64, 3, 2)
    x = residual_block(x, 64, 2, True)
    x = conv_block(x, 128, 3, 2)
    x = residual_block(x, 128, 2, True)
    x = conv_block(x, 256, 3, 2)
    x = residual_block(x, 256, 1, True)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Función para cargar imágenes y etiquetas
def load_images_and_labels(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (133, 133))
                images.append(img)
                label = float(filename.split('_')[0])  # Extraer la línea del metro
                labels.append(label)
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

input_folder = 'iconos_ruido'
images, labels = load_images_and_labels(input_folder)

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Número total de estaciones únicas
num_classes = len(np.unique(labels))

# Crear y compilar el modelo
model = build_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.save("clasificador_estaciones.h5")