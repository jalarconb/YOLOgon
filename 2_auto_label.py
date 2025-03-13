import cv2
import os
from mtcnn import MTCNN


def detect_and_label(image_folder, label_folder, class_id):
    detector = MTCNN()
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            height, width, _ = img.shape

            detections = detector.detect_faces(img)
            label_file = os.path.join(
                label_folder,
                filename.replace(".jpg", ".txt")
                .replace(".png", ".txt")
                .replace(".jpeg", ".txt"),
            )

            with open(label_file, "w") as f:
                for detection in detections:
                    x, y, w, h = detection["box"]
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

            print(f"Anotación generada para: {filename}")


if __name__ == "__main__":
    detect_and_label(
        "raw_data/images/with_glasses", "raw_data/labels/with_glasses", class_id=1
    )
    detect_and_label(
        "raw_data/images/no_glasses", "raw_data/labels/no_glasses", class_id=0
    )
    print("Proceso de anotación automática completado.")
