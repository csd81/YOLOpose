import cv2
import pyautogui
from ultralytics import YOLO

# --- PyAutoGUI Setup ---
pyautogui.FAILSAFE = False # Globally disabled to prevent corner crashes
pyautogui.PAUSE = 0        # Remove default delay for immediate response

# --- Configuration ---
DEADZONE = 50           # Radius of the "neutral" zone in pixels
SPEED_FACTOR = 0.3      # Sensitivity of the mouse movement
SMOOTHING = 0.7         # 0.0 = no smoothing, 0.9 = heavy smoothing (gliding)

IS_CLICKING = False     # Debounce flag for the left hand
smooth_vx, smooth_vy = 0.0, 0.0  # Variables to hold our smoothed velocity

# Load YOLO-Pose model
model = YOLO('yolov8n-pose.pt') 

# Start Webcam
cap = cv2.VideoCapture(0)

# Get camera resolution to calculate the center
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = cam_width // 2, cam_height // 2

print("Starting camera... Press 'q' to quit.")
print("Move Right Hand to move mouse. Raise Left Hand above Left Eye to click.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # Run YOLO pose inference
    results = model(frame, verbose=False)

    # Check if a person is detected and keypoints are available
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        
        kp = results[0].keypoints.xy[0].cpu().numpy()
        
        if len(kp) >= 11:
            left_eye = kp[1]
            left_wrist = kp[9]
            right_wrist = kp[10]
            
            # --- 1. MOUSE MOVEMENT (Right Hand) ---
            if right_wrist[0] > 0 and right_wrist[1] > 0:
                dx = right_wrist[0] - center_x
                dy = right_wrist[1] - center_y
                
                raw_move_x, raw_move_y = 0, 0
                
                # Calculate raw target velocity if outside deadzone
                if abs(dx) > DEADZONE:
                    raw_move_x = dx * SPEED_FACTOR
                if abs(dy) > DEADZONE:
                    raw_move_y = dy * SPEED_FACTOR
                    
                # --- APPLY SMOOTHING FILTER ---
                # Blends the previous frame's speed with the current target speed
                smooth_vx = (SMOOTHING * smooth_vx) + ((1.0 - SMOOTHING) * raw_move_x)
                smooth_vy = (SMOOTHING * smooth_vy) + ((1.0 - SMOOTHING) * raw_move_y)

                # Move the mouse using the smoothed integer values
                if int(smooth_vx) != 0 or int(smooth_vy) != 0:
                    pyautogui.move(int(smooth_vx), int(smooth_vy))

                # Visual: Draw line from center to right wrist
                cv2.line(frame, (center_x, center_y), (int(right_wrist[0]), int(right_wrist[1])), (255, 0, 0), 2)
                cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 8, (255, 0, 0), -1)

            # --- 2. MOUSE CLICK (Left Hand) ---
            if left_wrist[0] > 0 and left_eye[0] > 0:
                
                # Check if wrist is above the eye
                if left_wrist[1] < left_eye[1]:
                    if not IS_CLICKING:
                        pyautogui.click()
                        print("Click!")
                        IS_CLICKING = True
                        
                    # Visual: Highlight left wrist GREEN
                    cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 12, (0, 255, 0), -1)
                else:
                    IS_CLICKING = False
                    # Visual: Highlight left wrist RED
                    cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 8, (0, 0, 255), -1)

    # Visuals: Draw center point and deadzone
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), DEADZONE, (0, 255, 255), 1)

    cv2.imshow("YOLO Mouse Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()