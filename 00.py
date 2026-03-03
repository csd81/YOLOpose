import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 Pose model (nano version is fastest)
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    # verbose=False keeps the terminal clean
    
    results = model(frame, verbose=False, conf=0.5, device=0)

    for r in results:
        # Check if any person was detected
        if r.keypoints is not None:
            # Get keypoints as a tensor/array
            # x, y are at index 0 and 1
            # Index 5 is Left Shoulder, Index 0 is Nose
            kpts = r.keypoints.xyn[0] 

            if len(kpts) > 5:
                nose = kpts[0]
                l_shoulder = kpts[5]

                # If Nose Y-coordinate is significantly higher/lower than shoulder
                # Or if the horizontal distance between nose and shoulder is too large
                # (Note: normalized coordinates 0.0 to 1.0)
                
                status = "Good Posture"
                color = (0, 255, 0) # Green

                # Logic: If nose drops too close to shoulder level (slouching)
                if nose[1] > l_shoulder[1] - 0.1: 
                    status = "SIT STRAIGHT!"
                    color = (0, 0, 255) # Red

                # Draw a line between nose and shoulder for visual feedback
                cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLOv8 Posture Corrector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()