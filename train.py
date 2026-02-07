from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/dataset/DataSet/data.yaml',
    epochs=200,
    imgsz=640
)