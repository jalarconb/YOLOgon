import cv2
import torch
import os

def load_yolo_model(model_name='yolov5s'):
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    return model

def detect_glasses(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Unable to start the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR when trying to capture the video box")
            break
        
        # Realizar detección con YOLO
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        for _, row in detections.iterrows():
            x1, y1, x2, y2, confidence, name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            
            if "person" in name:  # Solo procesar detección de rostro
                label = "ON" if confidence > 0.5 else "OFF"
                color = (0, 255, 0) if label == "ON" else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Glasses Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    yolo_model = load_yolo_model()
    detect_glasses(yolo_model)
