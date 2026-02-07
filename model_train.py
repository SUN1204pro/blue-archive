import cv2
from ultralytics import YOLO


model = YOLO('/Users/sunix/Downloads/best.pt') 

video_path = "video_blue_archive.mp4" 


results = model.predict(source=video_path, show=True, save=True, conf=0.5)