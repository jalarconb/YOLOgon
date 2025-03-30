from ultralytics import YOLO

# Load YOLOv8n
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="train_data/data.yaml", epochs=10, imgsz=640, batch=8)

# Save the model
model.save("yolo_glasses.pt")

print("YOLO: Training complete and model saved.")

# ------------------------

# from ultralytics import settings
# settings.update({'datasets_dir': '.'})  # Forces YOLO to use the current directory as the dataset directory
