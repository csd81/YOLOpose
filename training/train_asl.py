from ultralytics import YOLO

# 1. Load the Nano model (smallest and fastest for your laptop)
model = YOLO('yolov8n.pt') 

# 2. Train the model
# imgsz=640 is standard, batch=16 fits well in 4GB VRAM
model.train(
    data='asl_data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,   # Force use of your Quadro P2000
    name='asl_model'
)