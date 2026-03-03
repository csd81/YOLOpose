import cv2
from ultralytics import YOLO
import time

# Load model and move to Quadro P2000
model = YOLO('yolov8n-pose.pt') 

# Timer variables
bad_posture_start_time = None
alert_threshold = 3  # Seconds of slouching before alarm triggers

cap = cv2.VideoCapture(0)
window_name = "Quadro-Powered Posture Monitor"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Inference using your GPU (device=0)
    results = model(frame, verbose=False, conf=0.5, device=0)
    
    # 1. CREATE THE FRAME FIRST so we can draw on it
    annotated_frame = results[0].plot()
    
    # Initialize status defaults
    status = "Good Posture"
    color = (0, 255, 0)

    # 2. Check if results and keypoints exist
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        
        kpts = results[0].keypoints.data[0] 
        
        if kpts[0][2] > 0.5 and (kpts[5][2] > 0.5 or kpts[6][2] > 0.5):
            nose_y = kpts[0][1].item()
            shoulder_y = kpts[5][1].item()

            # Calculate the actual gap
            current_gap = shoulder_y - nose_y
            
            # Now we can safely use annotated_frame
            cv2.putText(annotated_frame, f"Gap: {int(current_gap)}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # POSTURE LOGIC: Trigger if gap is too small
            # If the gap printed on screen is e.g. 40 when slouching, change 50 to 60
            if current_gap < 50: 
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                
                elapsed = time.time() - bad_posture_start_time
                
                if elapsed >= alert_threshold:
                    status = f"SIT UP! SLOUCHING: {int(elapsed)}s"
                    color = (0, 0, 255) # Red
            else:
                bad_posture_start_time = None

    # Draw Status Text
    cv2.putText(annotated_frame, status, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow(window_name, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
