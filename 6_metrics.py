import time
import numpy as np
import cv2
import os
from ultralytics import YOLO
import tensorflow as tf


import subprocess
import sys

# Run pip script from Python
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels


# Image and label directories
val_images = "train_data/images/val"
val_labels = "train_data/labels/val"

# Load validation data
x_val, y_val = load_data(val_images, val_labels)

# Load models
model_cnn = tf.keras.models.load_model("cnn_glasses.h5")
model_yolo = YOLO("yolo_glasses.pt")

# Predict: CNN
y_pred_cnn = (model_cnn.predict(x_val) > 0.5).astype(int).flatten()
y_pred_cnn = y_pred_cnn[: len(y_val)]  # Make sure dimensions match

# Verify dimensions
print("Shapes:")
print("y_val:", y_val.shape)
print("y_pred_cnn:", y_pred_cnn.shape)

# Evaluate: CNN
acc_cnn = accuracy_score(y_val, y_pred_cnn)
prec_cnn = precision_score(y_val, y_pred_cnn)
rec_cnn = recall_score(y_val, y_pred_cnn)
f1_cnn = f1_score(y_val, y_pred_cnn)

# Evaluate: YOLO
y_pred_yolo = []
y_true = []

time_yolo_start = time.time()
for i, img in enumerate(x_val):
    results = model_yolo(img * 255)
    has_glasses = 0
    for result in results:
        if len(result.boxes) > 0:
            has_glasses = 1
            break
    y_pred_yolo.append(has_glasses)
    y_true.append(y_val[i])
time_yolo_end = time.time()

acc_yolo = accuracy_score(y_true, y_pred_yolo)
prec_yolo = precision_score(y_true, y_pred_yolo)
rec_yolo = recall_score(y_true, y_pred_yolo)
f1_yolo = f1_score(y_true, y_pred_yolo)
time_yolo = time_yolo_end - time_yolo_start

# Show results

print(
    "\n\n----------------------------------------------------------------------\n\n\n"
)

print("\nCNN Performance:")
print(
    f"Accuracy: {acc_cnn:.4f}, Precision: {prec_cnn:.4f}, Recall: {rec_cnn:.4f}, F1-score: {f1_cnn:.4f}"
)

print("\nYOLO Performance:")
print(
    f"Accuracy: {acc_yolo:.4f}, Precision: {prec_yolo:.4f}, Recall: {rec_yolo:.4f}, F1-score: {f1_yolo:.4f}"
)
print(f"Inference Time (YOLO): {time_yolo:.2f} sec")

print("\n")
