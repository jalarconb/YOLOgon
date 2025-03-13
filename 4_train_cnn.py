import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2


def load_data(image_dir, label_dir, img_size=(640, 640)):
    data = []
    labels = []
    image_files = os.listdir(image_dir)

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)

        label = 0  # Default: No glasses
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    label = 1  # Glasses detected

        data.append(img)
        labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels


# Directorios de imágenes y etiquetas
train_images = "train_data/images/train"
train_labels = "train_data/labels/train"
val_images = "train_data/images/val"
val_labels = "train_data/labels/val"

# Cargar datos
x_train, y_train = load_data(train_images, train_labels)
x_val, y_val = load_data(val_images, val_labels)

# Definir modelo CNN
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(640, 640, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compilar modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar modelo
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=8)

# Guardar modelo
model.save("cnn_glasses.h5")

print("Entrenamiento completado. Modelo guardado como cnn_glasses.h5")
