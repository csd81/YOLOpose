import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

model = YOLO('yolov8n-pose.pt')

# --- Temporal History (15 frame visszatekintes) ---
HISTORY_LEN = 15

norm_dist_history  = deque(maxlen=HISTORY_LEN)
left_wrist_history  = deque(maxlen=HISTORY_LEN)
right_wrist_history = deque(maxlen=HISTORY_LEN)

# --- Kuszobertekek (elso futasnal hangold be!) ---
# Mekkora NormDist-valtozas szamit "tavolodasnak" 15 frame alatt
SPREAD_DELTA_THRESHOLD = 0.25
# Mekkora normalizalt sebesség szamit "mozgasnak" a karmester-kéznel
CONDUCTOR_VEL_THRESHOLD = 0.25  # vallszelesseg-egyseg / 15 frame

cap = cv2.VideoCapture(0)
window_name = "Gesture Lab v1  |  q = kilepes"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False, conf=0.5, device=0)
    annotated_frame = results[0].plot()

    spread_label    = "---"
    conductor_label = "---"
    debug_lines     = []

    if (len(results) > 0
            and results[0].keypoints is not None
            and len(results[0].keypoints.data) > 0):

        kpts = results[0].keypoints.data[0]

        # Segédfüggvények a keypoint adatok kiolvasásához
        def c(i):  return kpts[i][2].item()            # confidence
        def xy(i): return kpts[i][0].item(), kpts[i][1].item()  # (x, y)

        # ------------------------------------------------------------------ #
        # 1. NORMALIZALAS – vallszelesseg mint alap                           #
        # ------------------------------------------------------------------ #
        if c(5) > 0.5 and c(6) > 0.5:
            ls = np.array(xy(5))   # bal vall (left shoulder)
            rs = np.array(xy(6))   # jobb vall (right shoulder)
            shoulder_dist = np.linalg.norm(ls - rs)

            if shoulder_dist > 0 and c(9) > 0.5 and c(10) > 0.5:
                lw = np.array(xy(9))   # bal csuklo
                rw = np.array(xy(10))  # jobb csuklo

                wrist_dist = np.linalg.norm(lw - rw)
                norm_dist  = wrist_dist / shoulder_dist

                # Torteneti pufferek frissitese
                norm_dist_history.append(norm_dist)
                left_wrist_history.append(lw)
                right_wrist_history.append(rw)

                debug_lines.append(f"NormDist: {norm_dist:.2f}  (vállszélesség-egység)")

                # ----------------------------------------------------------#
                # 2A. TAVOLODIK / KOZELEDIK – NormDist idöbeli valtozasa    #
                # ----------------------------------------------------------#
                if len(norm_dist_history) == HISTORY_LEN:
                    delta = norm_dist_history[-1] - norm_dist_history[0]
                    debug_lines.append(f"Delta({HISTORY_LEN}f): {delta:+.3f}")

                    if delta > SPREAD_DELTA_THRESHOLD:
                        spread_label = ">> TAVOLODIK <<"
                    elif delta < -SPREAD_DELTA_THRESHOLD:
                        spread_label = "<< KOZELEDIK >>"
                    else:
                        spread_label = "Semleges"

                # ----------------------------------------------------------#
                # 2B. KARMESTER gesztus                                      #
                #   Master kez: tartosan az orr felett (y_wrist < y_nose)    #
                #   Slave kez: sebesség-vektor alapjan irany detektálás      #
                # ----------------------------------------------------------#
                if c(0) > 0.5:
                    nose_y = xy(0)[1]
                    # Kepkoordinataban Y felfelé no, ezert < = magasabban van
                    left_raised  = xy(9)[1]  < nose_y
                    right_raised = xy(10)[1] < nose_y

                    if left_raised and not right_raised:
                        master_side = "BAL"
                        hist = right_wrist_history
                        side_label = "Jobb csuklo"
                    elif right_raised and not left_raised:
                        master_side = "JOBB"
                        hist = left_wrist_history
                        side_label = "Bal csuklo"
                    else:
                        master_side = None
                        hist = None
                        side_label = None

                    if master_side and hist and len(hist) == HISTORY_LEN:
                        # Sebesség (normalizalva vallszelesseggel)
                        dx = (hist[-1][0] - hist[0][0]) / shoulder_dist
                        dy = (hist[-1][1] - hist[0][1]) / shoulder_dist
                        debug_lines.append(
                            f"{side_label}  vx:{dx:+.2f}  vy:{dy:+.2f}  (normált)"
                        )

                        if abs(dx) > CONDUCTOR_VEL_THRESHOLD or abs(dy) > CONDUCTOR_VEL_THRESHOLD:
                            if abs(dx) >= abs(dy):
                                conductor_label = "JOBBRA -->" if dx > 0 else "<-- BALRA"
                            else:
                                conductor_label = "v LE v" if dy > 0 else "^ FEL ^"
                        else:
                            conductor_label = f"[{master_side} fent – vár mozgásra]"

    # ------------------------------------------------------------------ #
    # HUD kirajzolása                                                      #
    # ------------------------------------------------------------------ #
    cv2.putText(annotated_frame, f"Karok:     {spread_label}",    (30,  50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Karmester: {conductor_label}", (30,  90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)

    for i, line in enumerate(debug_lines):
        cv2.putText(annotated_frame, line, (30, 140 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow(window_name, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
