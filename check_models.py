from ultralytics import YOLO
import torch

models = {
    "yolo26m_ppe":  "runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",
    "yolov8m_ppe":  "runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",
    "yolov9m_ppe":  "runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "yolov11m_ppe": "runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
}

for name, path in models.items():
    print(f"\n{'='*50}")
    print(f"Checking: {name}")
    try:
        model = YOLO(path)
        dummy = torch.zeros(1, 3, 640, 640)
        results = model.predict(source=dummy, imgsz=640, verbose=False)
        print(f"  OK - Loaded successfully")
        print(f"  Model type   : {type(model.model).__name__}")
        print(f"  Classes ({len(model.names)}): {list(model.names.values())}")
        print(f"  Output boxes : {len(results[0].boxes)}")
    except Exception as e:
        print(f"  FAILED: {e}")
