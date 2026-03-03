 

# YOLOv8 Posture Corrector 🧘‍♂️💻

A real-time posture monitoring application powered by **YOLOv8-Pose** and **NVIDIA Quadro P2000 GPU** (Pascal architecture). This tool monitors the vertical gap between your nose and shoulders to detect slouching and provides visual alerts after a configurable time threshold.

## 🚀 Key Features

* **GPU Accelerated:** Optimized for older NVIDIA Pascal GPUs (Quadro P2000) using custom PyTorch builds.
* **Debounce Timer:** Only alerts you if bad posture is maintained for more than 3 seconds.
* **Real-time Calibration:** Displays a live "Gap" value to help you fine-tune detection for your specific seating arrangement.
* **Automatic Cleanup:** Safely handles window closure and releases GPU resources.

---

## 🛠 Hardware & Software Stack

* **OS:** Ubuntu (Questing)
* **GPU:** NVIDIA Quadro P2000 (Mobile)
* **Python:** 3.13
* **AI Engine:** YOLOv8 (Ultralytics)
* **Compute:** CUDA 11.8 (via PyTorch 2.7.1+cu118)

---

## 📦 Installation

### 1. System Dependencies

You need specific OpenGL and Glib libraries for OpenCV to render the GUI on Ubuntu:

```bash
sudo apt update && sudo apt install -y libgl1 libglib2.0-0 fonts-dejavu-core

```

### 2. Environment Setup

Create a virtual environment to keep your system clean:

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. The "Magic Combo" PyTorch Install

Because the Quadro P2000 (sm_61) requires CUDA 11.8 compatibility on Python 3.13, install these specific wheels:

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python

```

---

## 🖥 Usage

Run the posture monitor:

```bash
python3 03.py

```

### Calibration Guide

1. **Observe the Gap:** Look at the "Gap" value displayed on the screen.
2. **Set Threshold:** If your good posture is ~150 and slouching is ~80, set the `threshold` in the code to **120**.
3. **Adjust Timer:** Change `alert_threshold` (default 3s) to suit your preference.

**Controls:**

* Press **'q'** or click the **'X'** on the window to quit.

---

## 📂 Project Structure

```text
.
├── 03.py              # Main application script
├── requirements.txt   # Exported stable dependencies
├── .gitignore         # Prevents venv and .pt models from being tracked
└── README.md          # You are here!

```

---

## ⚠️ Troubleshooting

* **IndexError:** The code includes safety checks (`if len(results) > 0`) to prevent crashes when you leave the camera frame.
* **CUDA Compatibility:** If you see "No kernel image available," ensure you are using the `+cu118` version of Torch.
* **Font Warnings:** If the terminal shows `QFontDatabase` errors, they are safely ignored; the app will still function.

---

### Would you like me to add a "How it Works" section with a diagram of the Nose-to-Shoulder math?