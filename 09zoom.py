import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
RESOLUTION        = (1280, 720)
MODEL_SIZE        = 'n'
CONF              = 0.4
IOU               = 0.45
DEVICE            = 0

HISTORY_LEN       = 20            # frames kept in rolling window

# Zoom thresholds on horizontal wrist distance / shoulder distance
ZOOM_IN_THRESH    = 1.8           # hands wider than this → keep zooming in
ZOOM_OUT_THRESH   = 0.6           # hands closer than this → keep zooming out
ALIGN_THRESH      = 0.5           # max vertical offset between wrists (norm) to count as horizontal
SLIDER_SPEED      = 0.01         # slider change per frame

# Brightness: both hands must be above shoulders to activate
# Above eye line → brighten, between eyes and shoulders → darken

# Zoom range applied to the live feed
ZOOM_MIN          = 0.5
ZOOM_MAX          = 2.0

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = YOLO(f'yolo26{MODEL_SIZE}-pose.pt')


# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
norm_dist_history   = deque(maxlen=HISTORY_LEN)

zoom_value   = 1/3   # maps to 1.0x (camera resolution)
bright_value = 0.5   # 0.0 → dark,  1.0 → bright  (0.5 = neutral)
invert_on    = False  # toggled by knee raise
knee_was_up  = False  # edge detection for toggle




# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def apply_brightness(frame, factor):
    """factor: 0..1 where 0.5 = unchanged, 0 = black, 1 = 2x bright."""
    scale = factor * 2.0   # 0→0, 0.5→1.0, 1.0→2.0
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)



def draw_vslider(frame, x, y, h, value, label, color):
    bw     = 22
    filled = int(h * value)
    cv2.rectangle(frame, (x, y),             (x + bw, y + h),          (40, 40, 40),    -1)
    cv2.rectangle(frame, (x, y + h - filled),(x + bw, y + h),           color,           -1)
    cv2.rectangle(frame, (x, y),             (x + bw, y + h),          (180, 180, 180),   1)
    cv2.putText(frame, label,         (x - 4, y - 8),         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,           1)
    cv2.putText(frame, f"{value:.2f}",(x - 4, y + h + 18),   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_hslider(frame, x, y, w, value, label, color):
    bh     = 22
    filled = int(w * value)
    cv2.rectangle(frame, (x,        y), (x + w,      y + bh), (40, 40, 40),    -1)
    cv2.rectangle(frame, (x,        y), (x + filled, y + bh),  color,           -1)
    cv2.rectangle(frame, (x,        y), (x + w,      y + bh), (180, 180, 180),   1)
    cv2.putText(frame, label,         (x, y - 8),              cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,           1)
    cv2.putText(frame, f"{value:.2f}",(x + w + 8, y + 15),    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_badge(frame, text, color, y):
    pad = 8
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.rectangle(frame, (18, y - th - pad), (18 + tw + pad * 2, y + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (18 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

WIN = "YOLO26 Gesture Control  |  q = quit"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WIN, 0, 0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results  = model(frame, verbose=False, conf=CONF, iou=IOU, device=DEVICE)
    canvas   = results[0].plot(boxes=False)

    if (len(results) > 0
            and results[0].keypoints is not None
            and len(results[0].keypoints.data) > 0):

        kpts = results[0].keypoints.data[0]

        def c(i):   return kpts[i][2].item()
        def xy(i):  return kpts[i][0].item(), kpts[i][1].item()

        # ── Normalization ──────────────────────────────────────────────────
        if c(5) > 0.5 and c(6) > 0.5:
            ls = np.array(xy(5))   # left  shoulder (kpt 5)
            rs = np.array(xy(6))   # right shoulder (kpt 6)

            # quit if shoulders cross (person turned around)
            if rs[0] > ls[0]:
                break

            shoulder_dist = np.linalg.norm(ls - rs)

            # ── Hands ────────────────────────────────────────────────────
            if shoulder_dist > 0 and c(9) > 0.5 and c(10) > 0.5:
                lw = np.array(xy(9))    # left  wrist (kpt 9)
                rw = np.array(xy(10))   # right wrist (kpt 10)

                norm_hdist = abs(lw[0] - rw[0]) / shoulder_dist   # horizontal
                norm_vdiff = abs(lw[1] - rw[1]) / shoulder_dist   # vertical offset between hands
                norm_dist_history.append(norm_hdist)

                # ── Zoom (only when hands are horizontally aligned) ───────
                if norm_vdiff < ALIGN_THRESH:
                    if norm_hdist >= ZOOM_IN_THRESH:
                        zoom_value = min(1.0, zoom_value + SLIDER_SPEED)
                    elif norm_hdist <= ZOOM_OUT_THRESH:
                        zoom_value = max(0.0, zoom_value - SLIDER_SPEED)

                # ── Reset if both hands below waistline ──────────────────
                if c(11) > 0.5 and c(12) > 0.5:
                    hip_mid_y = (xy(11)[1] + xy(12)[1]) / 2.0
                    if lw[1] > hip_mid_y and rw[1] > hip_mid_y:
                        zoom_value   = 1/3
                        bright_value = 0.5
                        norm_dist_history.clear()

                # ── Brightness (both hands above shoulders to activate) ──
                shoulder_mid_y = (ls[1] + rs[1]) / 2.0
                if lw[1] < shoulder_mid_y and rw[1] < shoulder_mid_y:
                    if c(1) > 0.5 and c(2) > 0.5:
                        eye_y = (xy(1)[1] + xy(2)[1]) / 2.0
                        avg_hand_y = (lw[1] + rw[1]) / 2.0
                        if avg_hand_y < eye_y:       # above eyes → brighten
                            bright_value = min(1.0, bright_value + SLIDER_SPEED)
                        else:                         # between eyes and shoulders → darken
                            bright_value = max(0.0, bright_value - SLIDER_SPEED)

    # ── Toggle invert on knee raise (edge-triggered) ─────────────────────
    knee_up = False
    if (len(results) > 0
            and results[0].keypoints is not None
            and len(results[0].keypoints.data) > 0):
        kpts = results[0].keypoints.data[0]
        def _c(i):  return kpts[i][2].item()
        def _y(i):  return kpts[i][1].item()
        if _c(11) > 0.5 and _c(12) > 0.5:
            hip_y = (_y(11) + _y(12)) / 2.0
            if (_c(13) > 0.5 and _y(13) < hip_y) or (_c(14) > 0.5 and _y(14) < hip_y):
                knee_up = True
    if knee_up and not knee_was_up:
        invert_on = not invert_on
    knee_was_up = knee_up

    # ── Apply brightness, invert & zoom to live feed ─────────────────────
    canvas      = apply_brightness(canvas, bright_value)
    if invert_on:
        canvas = cv2.bitwise_not(canvas)
    actual_zoom = ZOOM_MIN + zoom_value * (ZOOM_MAX - ZOOM_MIN)

    fh, fw = canvas.shape[:2]

    # ── Gesture badges ─────────────────────────────────────────────────────
    draw_badge(canvas, f'Zoom: {zoom_value:.2f}',       (50, 255, 100), 48)
    draw_badge(canvas, f'Bright: {bright_value:.2f}',   (255, 220, 50), 96)


    # ── Sliders ────────────────────────────────────────────────────────────
    # Zoom – vertical, right edge
    draw_vslider(canvas, fw - 70,  50, 220, zoom_value, 'ZOOM',  (50, 255, 100))
    cv2.putText(canvas, f"{actual_zoom:.1f}x",
                (fw - 74, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 100), 1)

    # Brightness – vertical, left-of-zoom
    draw_vslider(canvas, fw - 110, 50, 220, bright_value, 'BRT', (255, 220, 50))

    # ── Debug: NormDist ────────────────────────────────────────────────────
    if norm_dist_history:
        nd = norm_dist_history[-1]
        cv2.putText(canvas, f"NormDist: {nd:.2f}",
                    (60, fh - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── Scale window by zoom factor ─────────────────────────────────────
    if actual_zoom != 1.0:
        canvas = cv2.resize(canvas, (int(fw * actual_zoom), int(fh * actual_zoom)),
                            interpolation=cv2.INTER_LINEAR)

    cv2.imshow(WIN, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
