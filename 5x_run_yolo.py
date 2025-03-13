import cv2

# import torch
# import os
from ultralytics import YOLO


# def train_yolo(dataset_path="raw_data", epochs=10):
#     print("Entrenando modelo YOLO con el dataset...")
#     model = YOLO("yolov5s")
#     model.train(data=dataset_path, epochs=epochs)
#     model.save("yolo_glasses.pt")
#     print("Entrenamiento completado. Modelo guardado como yolo_glasses.pt")
#     return model


def train_yolo(data_path="train_data/data.yaml", epochs=10):
    model = YOLO("yolov8n.pt")  # Load YOLOv8n
    model.train(data=data_path, epochs=epochs, imgsz=640, batch=8)  # Train the model
    model.save("yolo_glasses.pt")  # Save the model

    print("Training complete and model saved.")


def detect_glasses(model_path="yolo_glasses.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Unable to start the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR when trying to capture the video box")
            break

        # Detect glasses using the trained model (YOLO)
        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                label = "Con Gafas" if confidence > 0.5 else "Sin Gafas"
                color = (0, 255, 0) if label == "Con Gafas" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        cv2.imshow("Detección de Gafas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # train_yolo()
    detect_glasses()
