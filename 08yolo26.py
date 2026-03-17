import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
RESOLUTION        = (2560, 1440)
MODEL_SIZE        = 'n'           # 'n' / 's' / 'm' / 'l' / 'x'
CONF              = 0.4
IOU               = 0.45
DEVICE            = 0

HISTORY_LEN       = 20            # frames kept in rolling window

# Zoom thresholds on horizontal wrist distance / shoulder distance
ZOOM_IN_THRESH    = 1.8           # hands wider than this → keep zooming in
ZOOM_OUT_THRESH   = 0.6           # hands closer than this → keep zooming out
ALIGN_THRESH      = 0.5           # max vertical offset between wrists (norm) to count as horizontal
SLIDER_SPEED      = 0.015         # slider change per frame

# Brightness: both hands must be above shoulders to activate
# Above eye line → brighten, between eyes and shoulders → darken

# Distance detection (body size in frame)
DIST_SMOOTH       = 10            # rolling average window for distance

# Zoom range applied to the live feed
ZOOM_MIN          = 1.0
ZOOM_MAX          = 3.0

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = YOLO(f'yolo26{MODEL_SIZE}-pose.pt')

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
norm_dist_history   = deque(maxlen=HISTORY_LEN)
body_size_history   = deque(maxlen=DIST_SMOOTH)

zoom_value   = 0.0   # 0.0 → ZOOM_MIN,  1.0 → ZOOM_MAX
bright_value = 0.5   # 0.0 → dark,  1.0 → bright  (0.5 = neutral)
dist_value   = 0.0   # 0.0 → far,  1.0 → close




# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def apply_brightness(frame, factor):
    """factor: 0..1 where 0.5 = unchanged, 0 = black, 1 = 2x bright."""
    scale = factor * 2.0   # 0→0, 0.5→1.0, 1.0→2.0
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)


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

WIN = "YOLO26 Gesture Control  |  q = quit"

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results  = model(frame, verbose=False, conf=CONF, iou=IOU, device=DEVICE)
    canvas   = results[0].plot()

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

            # ── Distance (body size in frame) ────────────────────────────
            frame_w = frame.shape[1]
            body_ratio = shoulder_dist / frame_w  # bigger = closer
            body_size_history.append(body_ratio)
            dist_value = np.clip(sum(body_size_history) / len(body_size_history), 0.0, 1.0)

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

                # ── Brightness (both hands above shoulders to activate) ──
                shoulder_mid_y = (ls[1] + rs[1]) / 2.0
                # both wrists must be above shoulders (smaller Y = higher)
                if lw[1] < shoulder_mid_y and rw[1] < shoulder_mid_y:
                    # use eye line as boundary (kpts 1=left eye, 2=right eye)
                    if c(1) > 0.5 and c(2) > 0.5:
                        eye_y = (xy(1)[1] + xy(2)[1]) / 2.0
                        avg_hand_y = (lw[1] + rw[1]) / 2.0
                        if avg_hand_y < eye_y:       # above eyes → brighten
                            bright_value = min(1.0, bright_value + SLIDER_SPEED)
                        else:                         # between eyes and shoulders → darken
                            bright_value = max(0.0, bright_value - SLIDER_SPEED)

    # ── Apply brightness & zoom to live feed ─────────────────────────────
    canvas      = apply_brightness(canvas, bright_value)
    actual_zoom = ZOOM_MIN + zoom_value * (ZOOM_MAX - ZOOM_MIN)
    canvas      = apply_zoom(canvas, actual_zoom)

    fh, fw = canvas.shape[:2]

    # ── Gesture badges ─────────────────────────────────────────────────────
    draw_badge(canvas, f'Zoom: {zoom_value:.2f}',       (50, 255, 100), 48)
    draw_badge(canvas, f'Bright: {bright_value:.2f}',   (255, 220, 50), 96)
    draw_badge(canvas, f'Distance: {dist_value:.2f}',   (50, 200, 255), 144)

    # ── Sliders ────────────────────────────────────────────────────────────
    # Zoom – vertical, right edge
    draw_vslider(canvas, fw - 70,  50, 220, zoom_value, 'ZOOM',  (50, 255, 100))
    cv2.putText(canvas, f"{actual_zoom:.1f}x",
                (fw - 74, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 100), 1)

    # Brightness – vertical, left-of-zoom
    draw_vslider(canvas, fw - 110, 50, 220, bright_value, 'BRT', (255, 220, 50))

    # Distance – vertical, left edge
    draw_vslider(canvas, 28, 150, 220, dist_value, 'DIST', (50, 200, 255))

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
