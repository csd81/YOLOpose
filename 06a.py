import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
RESOLUTION        = (1280, 720)   # change to (640, 480) if camera can't handle 720p
MODEL_SIZE        = 'n'           # 'n' / 's' / 'm'
CONF              = 0.4
IOU               = 0.45
DEVICE            = 0

HISTORY_LEN       = 20            # frames kept in rolling window

# Spread / Close
MIN_FRAME_DELTA   = 0.008         # min per-frame change in norm_dist to count as movement
MAJORITY_FRAC     = 0.60          # fraction of frames that must show movement
MIN_TOTAL_DELTA   = 0.12          # min total change over full window (noise guard)

# Conductor
COND_HISTORY      = 6             # recent frames used for velocity
COND_VEL_THRESH   = 0.018         # normalised velocity threshold

# Slider update speed per frame (0 → 1 range, so 0.015 ≈ 1.5 s to traverse full bar)
SLIDER_SPEED      = 0.015

# Zoom range applied to the live feed
ZOOM_MIN          = 1.0
ZOOM_MAX          = 3.0

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = YOLO(f'yolov8{MODEL_SIZE}-pose.pt')

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
norm_dist_history   = deque(maxlen=HISTORY_LEN)
left_wrist_history  = deque(maxlen=HISTORY_LEN)
right_wrist_history = deque(maxlen=HISTORY_LEN)

zoom_value = 0.0   # 0.0 → ZOOM_MIN,  1.0 → ZOOM_MAX
h_value    = 0.5   # horizontal slider  (conductor L/R)
v_value    = 0.5   # vertical slider    (conductor U/D)


# ─────────────────────────────────────────────
#  GESTURE DETECTION
# ─────────────────────────────────────────────
def detect_spread(history):
    """
    Returns 'SPREADING', 'CLOSING', or None.

    Logic: at least MAJORITY_FRAC of consecutive per-frame deltas must exceed
    MIN_FRAME_DELTA in magnitude AND point the same direction, AND the total
    change over the window must exceed MIN_TOTAL_DELTA.
    """
    if len(history) < HISTORY_LEN:
        return None

    vals  = list(history)
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    n     = len(diffs)

    pos = sum(1 for d in diffs if d >  MIN_FRAME_DELTA)
    neg = sum(1 for d in diffs if d < -MIN_FRAME_DELTA)
    total_delta = vals[-1] - vals[0]

    if pos / n >= MAJORITY_FRAC and total_delta >  MIN_TOTAL_DELTA:
        return 'SPREADING'
    if neg / n >= MAJORITY_FRAC and total_delta < -MIN_TOTAL_DELTA:
        return 'CLOSING'
    return None


def detect_conductor(wrist_history, shoulder_dist):
    """
    Returns 'RIGHT', 'LEFT', 'UP', 'DOWN', 'STILL', or None.
    Uses the last COND_HISTORY frames for velocity.
    """
    if wrist_history is None or len(wrist_history) < COND_HISTORY:
        return None

    recent = list(wrist_history)[-COND_HISTORY:]
    dx = (recent[-1][0] - recent[0][0]) / shoulder_dist
    dy = (recent[-1][1] - recent[0][1]) / shoulder_dist

    if abs(dx) < COND_VEL_THRESH and abs(dy) < COND_VEL_THRESH:
        return 'STILL'
    if abs(dx) >= abs(dy):
        return 'RIGHT' if dx > 0 else 'LEFT'
    return 'DOWN' if dy > 0 else 'UP'


# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def apply_zoom(frame, zoom_factor):
    """Centre-crop then upscale → gives a zoom effect on the live feed."""
    h, w   = frame.shape[:2]
    scale  = 1.0 / zoom_factor
    nh, nw = int(h * scale), int(w * scale)
    y1, x1 = (h - nh) // 2, (w - nw) // 2
    return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h))


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

WIN = "Gesture Control  |  q = quit"

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results  = model(frame, verbose=False, conf=CONF, iou=IOU, device=DEVICE)
    canvas   = results[0].plot()

    spread_gesture    = None
    conductor_gesture = None
    master_side       = None
    slave_hist        = None

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
            shoulder_dist = np.linalg.norm(ls - rs)

            if shoulder_dist > 0 and c(9) > 0.5 and c(10) > 0.5:
                lw = np.array(xy(9))    # left  wrist (kpt 9)
                rw = np.array(xy(10))   # right wrist (kpt 10)

                norm_dist = np.linalg.norm(lw - rw) / shoulder_dist
                norm_dist_history.append(norm_dist)
                left_wrist_history.append(lw)
                right_wrist_history.append(rw)

                # ── Spread / Close ─────────────────────────────────────────
                spread_gesture = detect_spread(norm_dist_history)

                if spread_gesture == 'SPREADING':
                    zoom_value = min(1.0, zoom_value + SLIDER_SPEED)
                elif spread_gesture == 'CLOSING':
                    zoom_value = max(0.0, zoom_value - SLIDER_SPEED)

                # ── Conductor ──────────────────────────────────────────────
                if c(0) > 0.5:
                    nose_y = xy(0)[1]
                    l_raised = xy(9)[1]  < nose_y   # wrist above nose = raised
                    r_raised = xy(10)[1] < nose_y

                    if l_raised and not r_raised:
                        master_side = 'LEFT'
                        slave_hist  = right_wrist_history
                    elif r_raised and not l_raised:
                        master_side = 'RIGHT'
                        slave_hist  = left_wrist_history

                    if master_side:
                        conductor_gesture = detect_conductor(slave_hist, shoulder_dist)

                        if conductor_gesture == 'RIGHT':
                            h_value = min(1.0, h_value + SLIDER_SPEED)
                        elif conductor_gesture == 'LEFT':
                            h_value = max(0.0, h_value - SLIDER_SPEED)
                        elif conductor_gesture == 'UP':
                            v_value = max(0.0, v_value - SLIDER_SPEED)   # up = decrease Y
                        elif conductor_gesture == 'DOWN':
                            v_value = min(1.0, v_value + SLIDER_SPEED)

    # ── Apply zoom to live feed ────────────────────────────────────────────
    actual_zoom = ZOOM_MIN + zoom_value * (ZOOM_MAX - ZOOM_MIN)
    canvas      = apply_zoom(canvas, actual_zoom)

    fh, fw = canvas.shape[:2]

    # ── Gesture badges ─────────────────────────────────────────────────────
    spread_color = {
        'SPREADING': (50,  255, 100),
        'CLOSING':   (50,  100, 255),
        None:        (140, 140, 140),
    }[spread_gesture]

    spread_text = {
        'SPREADING': '>> SPREADING <<',
        'CLOSING':   '<< CLOSING >>',
        None:        'Arms: neutral',
    }[spread_gesture]

    if master_side is None:
        cond_text  = 'Conductor: --'
        cond_color = (140, 140, 140)
    elif conductor_gesture in (None, 'STILL'):
        cond_text  = f'Conductor [{master_side} raised – waiting]'
        cond_color = (200, 165,  50)
    else:
        arrows     = {'RIGHT': '-->', 'LEFT': '<--', 'UP': '^ UP', 'DOWN': 'v DOWN'}
        cond_text  = f'Conductor: {arrows[conductor_gesture]}'
        cond_color = (50, 200, 255)

    draw_badge(canvas, spread_text, spread_color, 48)
    draw_badge(canvas, cond_text,   cond_color,   96)

    # ── Sliders ────────────────────────────────────────────────────────────
    # Zoom – vertical, right edge
    draw_vslider(canvas, fw - 70,  50, 220, zoom_value, 'ZOOM',  (50, 255, 100))
    cv2.putText(canvas, f"{actual_zoom:.1f}x",
                (fw - 74, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 100), 1)

    # H-slider – horizontal, bottom
    draw_hslider(canvas, 60, fh - 55, fw - 180, h_value, 'H', (50, 200, 255))

    # V-slider – vertical, left edge (separate parameter)
    draw_vslider(canvas, 28, 150, 220, v_value, 'V', (255, 165, 50))

    # ── Debug: NormDist ────────────────────────────────────────────────────
    if norm_dist_history:
        nd = norm_dist_history[-1]
        cv2.putText(canvas, f"NormDist: {nd:.2f}",
                    (60, fh - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow(WIN, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
