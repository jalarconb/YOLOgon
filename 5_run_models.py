import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Cargar modelos
model_yolo = YOLO("yolo_glasses.pt")
model_cnn = tf.keras.models.load_model("cnn_glasses.h5")

# Dimensión esperada para la CNN
IMG_SIZE = (128, 128)


def preprocess_cnn(frame):
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


# Capturar video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con YOLO
    results = model_yolo(frame)
    yolo_prediction = "No glasses"
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "Glasses" if int(box.cls[0]) == 1 else "No glasses"
            yolo_prediction = label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Detección con CNN
    cnn_input = preprocess_cnn(frame)
    cnn_prediction = (
        "Glasses" if model_cnn.predict(cnn_input)[0][0] > 0.5 else "No glasses"
    )

    # Dibujar rectángulo en toda la imagen si la CNN predice gafas
    if cnn_prediction == "Glasses":
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)  # Rectángulo rojo

    # Mostrar resultados en pantalla
    cv2.putText(
        frame,
        f"YOLO: {yolo_prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"CNN: {cnn_prediction}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Glasses Detection Comparison", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
