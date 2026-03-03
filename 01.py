import cv2
from ultralytics import YOLO
import time

# Load model and move to Quadro P2000
model = YOLO('yolov8n-pose.pt') 

# Timer variables
bad_posture_start_time = None
alert_threshold = 3  # Seconds of slouching before alarm triggers

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Inference using your GPU (device=0)
    results = model(frame, verbose=False, conf=0.5, device=0)
    
    # Initialize status defaults
    status = "Good Posture"
    color = (0, 255, 0)

    # 1. FIX THE CRASH: Check if results and keypoints exist
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        
        # Get coordinates for first person detected
        # YOLOv8 keypoint 0 = Nose, 5 = Left Shoulder, 6 = Right Shoulder
        kpts = results[0].keypoints.data[0] 
        
        # Ensure we have high confidence for the Nose and at least one Shoulder
        if kpts[0][2] > 0.5 and (kpts[5][2] > 0.5 or kpts[6][2] > 0.5):
            nose_y = kpts[0][1].item()
            shoulder_y = kpts[5][1].item()

            # 2. POSTURE LOGIC: Slouching usually means Nose Y is too close to Shoulder Y
            # Adjust the '- 50' based on how far you sit from the camera
            if nose_y > (shoulder_y - 50):
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                
                elapsed = time.time() - bad_posture_start_time
                
                if elapsed >= alert_threshold:
                    status = f"SIT UP! SLOUCHING: {int(elapsed)}s"
                    color = (0, 0, 255) # Red
            else:
                # Reset timer if posture is corrected
                bad_posture_start_time = None

    # Visualize skeleton
    annotated_frame = results[0].plot()
    
    # Draw Status Text
    cv2.putText(annotated_frame, status, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Quadro-Powered Posture Monitor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()