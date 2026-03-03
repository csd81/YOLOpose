# A valós idejű kéz-kulcspont becslés technológiai fejlődése: A YOLO11 és a MediaPipe keretrendszerek összehasonlító elemzése

A számítógépes látás területe az elmúlt évtizedben radikális átalakuláson ment keresztül, elmozdulva az egyszerű tárgyfelismeréstől a komplex emberi viselkedés és finommotoros mozgások precíz digitalizálása felé. A kéz-kulcspont becslés (Hand Keypoint Estimation) ezen evolúció egyik legkritikusabb szegmense, amely az ember-gép interakció (HCI) új dimenzióit nyitja meg. Ez a technológia nem csupán a kéz pozícióját határozza meg a térben, hanem annak anatómiai felépítését is leképezi egy 21 pontból álló vázmodell segítségével, lehetővé téve a gesztusok, a jeltolmácsolás és a virtuális objektumokkal való manipuláció pontos nyomon követését.

A modern keretrendszerek, mint például az Ultralytics YOLO11 és a Google MediaPipe, eltérő építészeti filozófiát követnek a feladat megoldása érdekében, miközben mindkettő a valós idejű teljesítmény és a nagy pontosság közötti optimális egyensúly megteremtésére törekszik.

---

## A kéz-kulcspont becslés anatómiai és matematikai alapjai

A kéz digitális reprezentációja egy szigorúan meghatározott hierarchikus modellen alapul. A legtöbb modern adatkészlet és algoritmus, beleértve az Ultralytics által támogatott formátumokat is, 21 egyedi kulcspontot azonosít, amelyek a csuklótól az ujjhegyekig terjednek. Ez a topológia lehetővé teszi a kéz szabadságfokainak (Degrees of Freedom – DoF) közel teljes modellezését, ami elengedhetetlen a komplex gesztusok és finommotoros mozdulatok felismeréséhez.

### A 21 kulcspontos modell felépítése

A kulcspontok elrendezése követi az emberi kéz fiziológiai struktúráját. Minden ujj négy kulcsponttal rendelkezik:

* MCP (metacarpophalangeal) – ujj töve
* PIP (proximal interphalangeal) – középső ízület
* DIP (distal interphalangeal) – felső ízület
* Ujjhegy

A csukló (Wrist) szolgál a modell origójaként, amelyhez az összes ujj láncolatszerűen kapcsolódik.

| Kulcspont index | Anatómiai megnevezés | Ujjcsoport    |
| --------------- | -------------------- | ------------- |
| 0               | Csukló (Wrist)       | Tenyér alapja |
| 1               | Hüvelykujj MCP       | Hüvelykujj    |
| 2               | Hüvelykujj PIP       | Hüvelykujj    |
| 3               | Hüvelykujj DIP       | Hüvelykujj    |
| 4               | Hüvelykujj hegye     | Hüvelykujj    |
| 5               | Mutatóujj MCP        | Mutatóujj     |
| 6               | Mutatóujj PIP        | Mutatóujj     |
| 7               | Mutatóujj DIP        | Mutatóujj     |
| 8               | Mutatóujj hegye      | Mutatóujj     |
| 9               | Középső ujj MCP      | Középső ujj   |
| 10              | Középső ujj PIP      | Középső ujj   |
| 11              | Középső ujj DIP      | Középső ujj   |
| 12              | Középső ujj hegye    | Középső ujj   |
| 13              | Gyűrűsujj MCP        | Gyűrűsujj     |
| 14              | Gyűrűsujj PIP        | Gyűrűsujj     |
| 15              | Gyűrűsujj DIP        | Gyűrűsujj     |
| 16              | Gyűrűsujj hegye      | Gyűrűsujj     |
| 17              | Kisujj MCP           | Kisujj        |
| 18              | Kisujj PIP           | Kisujj        |
| 19              | Kisujj DIP           | Kisujj        |
| 20              | Kisujj hegye         | Kisujj        |

Ez a strukturált adatmodell lehetővé teszi vektorok, szögek és ízületi hajlítások számítását. Például két kulcspont közötti vektor:

$$
\vec{v}_{ij} = (x_j - x_i,; y_j - y_i)
$$

Három pont által meghatározott ízületi szög:

$$
\theta = \arccos \left( \frac{\vec{v}*{ij} \cdot \vec{v}*{ik}}{||\vec{v}*{ij}|| ; ||\vec{v}*{ik}||} \right)
$$

A YOLO11 és hasonló modellek esetében minden kulcspontot egy $(x, y)$ koordináta-pár és egy láthatósági jelző (visibility flag) határoz meg.

---

## Objektum-kulcspont hasonlóság (OKS) és hibaarányok

A pózbecslési modellek értékelése eltér a klasszikus osztályozási feladatoktól. Az Object Keypoint Similarity (OKS) figyelembe veszi az objektum méretét és az annotáció bizonytalanságát:

$$
OKS = \frac{\sum_{i} \exp(-d_i^2 / 2s^2 \kappa_i^2) \delta(v_i > 0)}{\sum_{i} \delta(v_i > 0)}
$$

ahol:

* $d_i$ – euklideszi távolság a predikció és ground truth között
* $s$ – objektum skálája
* $\kappa_i$ – kulcspont-specifikus normalizáló tényező
* $\delta(v_i > 0)$ – láthatósági indikátor

Az OKS alapú mAP (mean Average Precision) a standard mérőszám.

---

## Az Ultralytics YOLO11 ökoszisztéma és a pózbecslés integrációja

A YOLO11 az Ultralytics legújabb generációs architektúrája, amely többfeladatos tanulásra optimalizált.

### Architektúra

Fő komponensek:

* Backbone – jellemzők kinyerése
* Neck – több skálán történő aggregáció
* Decoupled head – külön ág bounding box, osztály és kulcspont regresszió számára

A C3k2 blokkok és a PSA (Position-aware Self-Attention) modulok globális kontextust biztosítanak.

---

## Adatfeldolgozás és Hand Keypoints Dataset

A COCO keypoint formátumon alapul. Tartalmaz:

* változatos megvilágítás
* különböző bőrtónusok
* több kézpozíció

### Adat-augmentáció

* Mosaic augmentation
* Keypoint flipping
* Affine transzformációk
* Színjitter, blur, zaj

---

## MediaPipe: Pipeline architektúra

Kétlépcsős rendszer:

1. BlazePalm – tenyér detektálás
2. Hand Landmark Model – 21 kulcspont regresszió

Előny: temporal tracking, detektor kihagyása stabil követés esetén.

### Futtatási módok

* IMAGE
* VIDEO
* LIVE_STREAM

WASM és TFLite támogatás.

---

## Összehasonlító elemzés

| Szempont     | YOLO11 Pose | MediaPipe Hands  |
| ------------ | ----------- | ---------------- |
| Hardver      | NVIDIA GPU  | CPU / Mobile GPU |
| Architektúra | End-to-End  | Pipeline         |
| Több kéz     | Párhuzamos  | Szekvenciális    |
| Modell méret | 15–100MB+   | 3–5MB            |
| Távoli kéz   | Erős        | Gyengébb         |

YOLO erőssége: kontextusérzékenység, occlusion-kezelés.
MediaPipe erőssége: alacsony késleltetés, stabil landmark.

---

## Modelloptimalizálás és export

| Formátum | Platform       | Előny             |
| -------- | -------------- | ----------------- |
| ONNX     | Cross-platform | Kompatibilitás    |
| TensorRT | NVIDIA GPU     | Maximális FPS     |
| CoreML   | Apple          | Neural Engine     |
| TFLite   | Android        | Alacsony memória  |
| OpenVINO | Intel          | CPU optimalizáció |

### Kvantálás

FP32 → INT8 konverzió.
Sebességnövekedés és memória-csökkentés minimális pontosságvesztéssel.

---

## Gyakorlati alkalmazások

### AR/VR

* Pinch detektálás
* Objektum manipuláció
* Alacsony latency kritikus

### Jeltolmácsolás

* Időbeli kulcspont-sorozatok elemzése
* Gesztus-szekvenciák modellezése (pl. LSTM, Transformer)

### Egészségügy

* Ízületi ROM mérés
* Rehabilitációs visszajelzés
* Távoli monitorozás

---

## Műszaki kihívások

### Okklúzió

Ön-takarás → implicit következtetés szükséges.

### 2D–3D kétértelműség

A MediaPipe relatív $z$ koordinátát becsül, de valódi mélység szenzor nélkül korlátozott.

### Megvilágítás

Adat-augmentáció és robusztus detektálás kulcsfontosságú.

---

## Jövőbeli irányok

* Vision Transformers
* Foundation modellek
* Multimodális rendszerek
* Szintetikus adatok (Unreal Engine)
* 3D supervision depth szenzorral

---

## Összegzés

A YOLO11 a nagy teljesítményű, GPU-alapú, skálázható rendszerek számára ideális, ahol több kéz és komplex jelenet feldolgozása szükséges. A MediaPipe mobil- és weborientált környezetben nyújt hatékony megoldást alacsony késleltetéssel és kis modellmérettel.

A technológia fejlődése a valós idejű, kontextusérzékeny, 3D-tudatos modellek irányába halad, amelyek a kézmozgást természetes interfésszé alakítják a digitális rendszerekben.
 

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