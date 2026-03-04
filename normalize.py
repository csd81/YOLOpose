import cv2
import numpy as np
from ultralytics import YOLO

# Load the smallest, fastest model as Richárd suggested (n = nano)
model = YOLO('yolov8n-pose.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference
    results = model(frame, verbose=False, conf=0.5)

    for r in results:
        if r.keypoints is not None:
            # Get keypoints as (x, y) coordinates
            # YOLO Keypoint Indices: 5,6 = Shoulders | 9,10 = Wrists
            kpts = r.keypoints.xy[0].cpu().numpy()

            try:
                # 1. Calculate Shoulder Width (The Normalization Base)
                shoulder_dist = np.linalg.norm(kpts[5] - kpts[6])
                
                # 2. Calculate Wrist Distance
                wrist_dist = np.linalg.norm(kpts[9] - kpts[10])

                # 3. Normalized Distance (Robus to distance from camera)
                # This is the "Decision Rule" Richárd wants
                norm_dist = wrist_dist / shoulder_dist if shoulder_dist > 0 else 0

                # Simple Decision Logic
                status = "Neutral"
                if norm_dist > 2.0:
                    status = "ARMS APART"
                elif norm_dist < 0.5:
                    status = "ARMS CLOSE"

                # Visual Feedback
                cv2.putText(frame, f"Norm Dist: {norm_dist:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {status}", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except IndexError:
                # Skip frame if keypoints aren't detected
                pass

    cv2.imshow("YOLOv8 Pose - Gesture Lab", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()