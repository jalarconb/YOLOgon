import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2


def load_data(image_dir, label_dir, img_size=(128, 128)):
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

    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)

    print("Distribución de clases en dataset:", np.bincount(labels))
    return data, labels


# Image and label directories
train_images = "train_data/images/train"
train_labels = "train_data/labels/train"
val_images = "train_data/images/val"
val_labels = "train_data/labels/val"

# Load data
x_train, y_train = load_data(train_images, train_labels)
x_val, y_val = load_data(val_images, val_labels)

# CNN Model (4*conv, 1*flatten, 1*output)
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),  # Lowers overfitting
        layers.Dense(1, activation="sigmoid"),  # Binary classification: Glasses or no glasses
    ]
)

# Compile model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Train Model
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)

# Save model
model.save("cnn_glasses.h5")

print("Training complete. Model saved as cnn_glasses.h5")
