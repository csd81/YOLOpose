# Valós idejű testpóz-alapú gesztúravezérlő rendszer fejlesztése YOLO pose-becsléssel

**VEMIINM336L – Kutatási/fejlesztési projekt beszámoló**
**Műszaki Informatikai Kar**
**Témavezető: Dr. Magyar Attila**
**Dátum: 2026. március 17.**

---

## Tartalomjegyzék

1. Bevezetés
2. Irodalom- és versenytárselemzés
3. Fejlesztési folyamat és saját eredmények
   - 3.1 Hardver- és szoftverkörnyezet felállítása
   - 3.2 Testtartás-figyelő prototípus (00.py – 05.py)
   - 3.3 Normalizált távolságszámítás és gesztusalap (normalize.py)
   - 3.4 Temporális gesztusfelismerés (06_gestures.py)
   - 3.5 Összetett gesztusvezérlő rendszer – 1. iteráció (06a.py)
   - 3.6 Gesztusvezérlő rendszer – 2. iteráció: zoom és fényerő (07_gesture_control.py)
   - 3.7 Átállás YOLO26-ra és inverz-toggle (08yolo26.py)
   - 3.8 Finomított zoom-logika és ablakméretezés (09zoom.py)
   - 3.9 RGB-csatorna-vezérlés és álló-detektálás (10rgb.py, 11detect_standing.py)
   - 3.10 Egérvezérlés testtartással (mouse.py)
   - 3.11 ASL-ábécé felismerés – tanítási kísérlet (training/)
4. A saját és korábbi eredmények összehasonlítása
5. Továbbfejlesztési lehetőségek
6. Összegzés
7. Irodalomjegyzék

---

## 1. Bevezetés

A számítógépes látás (computer vision) és a valós idejű emberi testpóz-becslés (human pose estimation) az elmúlt évtizedben hatalmas fejlődésen ment keresztül. A mélytanuláson alapuló detektáló modellek – különösen a You Only Look Once (YOLO) architektúra-família – lehetővé tették, hogy egyetlen kameraképből, egyetlen hálózati előremenő menetben detektáljuk és pontosan lokalizáljuk az emberi test anatómiai kulcspontjait (keypoints). Ez az infrastruktúra új dimenziót nyit az ember–gép interakció (Human–Computer Interaction, HCI) területén: az érintőfelület, az egér és a billentyűzet helyett maga a test válhat a felhasználói bemenet egyetlen eszközévé.

A jelen kutatási-fejlesztési projekt célja egy iteratívan bővített, valós idejű, kamera-alapú geszturavezérlő rendszer megtervezése és implementálása YOLO-alapú testpóz-becsléssel, NVIDIA GPU hardveres gyorsítással, Python és OpenCV eszköztárral. A fejlesztés egy egyszerű testtartás-figyelő prototípustól indult, és fokozatosan jutott el egy összetett, több gesztust párhuzamosan kezelő, kép-paramétereket (zoom, fényerő, RGB-csatornák) valós időben manipuláló vezérlőrendszerig, amelyet végül egy dedikált egérvezérlő modullal és egy ASL (American Sign Language) ábécé-felismerő tanítási kísérlettel egészítettem ki.

A fejlesztés során empirikus szemléletet alkalmaztam: minden egyes iteráció a korábbi verzió konkrét hiányosságainak azonosításán, majd célzott bővítésén alapult. A projekt tudományos hozzájárulása a következő területeken összpontosul:

- A vállszélesség-alapú normalizáció alkalmazása skálainvariáns gesztusfelismeréshez
- Temporális csúszó ablak (rolling window) és majoritásos döntési logika kombinálása zajszűréshez
- Testpóz-zóna-hierarchia (csípő, váll, szem szintje) mint szemantikai koordináta-rendszer
- A YOLO26 (Ultralytics YOLO v2.6 generáció) és a YOLOv8 összehasonlítása valós idejű gesztusvezérlő alkalmazásban
- Álló személy detektálása kulcspont-láthatósági feltételek alapján

---

## 2. Irodalom- és versenytárselemzés

### 2.1 Emberi testpóz-becslés – áttekintés

Az emberi testpóz-becslés (Human Pose Estimation, HPE) feladata, hogy egy képen vagy videóban meghatározza az emberi test anatomóiai kulcspontjainak (ízületek, végtagvégek) pozícióját. A területen az elmúlt évtizedben két megközelítés dominált: a bottum-up (előbb az összes kulcspontot detektálják, majd személyekhez rendelik) és a top-down (előbb az összes személyt detektálják, majd minden személyen belül a kulcspontokat) módszertan.

**OpenPose (Cao et al., 2017)** az egyik első, valóban valós idejű bottom-up HPE rendszer volt, amely Part Affinity Fields (PAF) segítségével kapcsolta össze a kulcspontokat. Az OpenPose referenciapont lett a területen, ám erős GPU-igénye és a viszonylag bonyolult telepítési lánca hátránynak bizonyult éles alkalmazásokban.

**MediaPipe Pose (Lugaresi et al., 2019, Google)** kétlépéses pipeline: egy könnyűsúlyú BlazePose detektor gyorsan megtalálja a személyt, majd egy regressziós modell az összes 33 kulcspontot becsüli egyidejűleg. A rendszer a CPU-n is valós idejű sebességgel fut, kifejezetten mobilra és webre optimalizált (TFLite, WASM). Hátránya, hogy elsősorban egyetlen személy szoros közelképére van optimalizálva, és a 33 pontos modell helyett a saját projektünkben értelmezett gesztusokhoz a 17 pontos COCO-topológia is elegendő.

**YOLOv8 Pose (Ultralytics, 2023)** az Ultralytics YOLO-ökoszisztéma pose-becsléssel kibővített változata. Az architektúra egy decoupled head-et alkalmaz: külön ágon regredálja a bounding boxokat, az osztálycímkéket és a kulcspontokat. A COCO-dataset 17 kulcspontos definícióját követi. Előnye, hogy GPU-n rendkívül gyors (akár >60 FPS a nano változattal), és több személy párhuzamos feldolgozására is alkalmas.

**YOLO11 / YOLO26 (Ultralytics, 2024–2025)** az előző generáció utódja. A C3k2 blokkok, PSA (Position-aware Self-Attention) modulok és az újratervezett neck-aggregáció együttesen javítják a pontosságot, különösen kis méretű és részlegesen takart kulcspontoknál. A jelen projekt később erre az architektúrára migrált (yolo26n/s/m/l/x-pose.pt).

### 2.2 Hasonló geszturavezérlő rendszerek

**HandsFree.js (2018, Morrison et al.)** egy böngésző-alapú geszturavezérlő könyvtár, amely a MediaPipe Face Mesh és Handpose modelljeit kombinálja. A fej és kéz mozgása alapján képes az egérkurzort vezérelni. Hátránya, hogy böngészőbe zárt, és nem képes teljes testpóz-alapú gesztúrák feldolgozására.

**GestureBot (2020, Microsoft Research)** egy kereskedelmi célú kutatási projekt, amely szobában elhelyezett RGB-D kamerák (Microsoft Kinect) segítségével vezérel otthoni eszközöket. A mélységinformáció (depth) kezeli az occlusion-problémát, amelyet a jelen projekt pusztán RGB-kamerával old meg. A Kinect 2013-ban piacra lépett, de 2017-ben megszűnt; azóta a csak-RGB megközelítések kerültek előtérbe.

**AirTouch (2021, Samsung Research)** érintés nélküli okosTV-vezérlés, amely a TV beépített kameráján futó hézagos kézmozdulat-detektálóval dolgozik. Kizárólag előre definiált, statikus kézgesztusokat ismer fel, temporális komponens nélkül. A mi rendszerünk ezzel szemben folyamatos, analóg értéket állít be (zoom, fényerő), nem bináris állapotokat.

**BodyPix + custom gesture (2022, TensorFlow.js)** egy példaprojekt, amely böngészőben futó BodyPix szegmentációt kombinál egyedi gesztusszabályokkal. A feldolgozás CPU-n ~10 FPS, ami valós idejű alkalmazáshoz nem elegendő. A jelen projekt GPU-n 60+ FPS-t ér el.

Összefoglalva: a hasonló megoldások vagy dedikált depth-szenzorokra támaszkodnak, vagy böngészőbe zártak, vagy csak statikus gesztusokat ismernek fel, vagy nem valós idejűek. A jelen projekt önálló hozzájárulása egy egységes, GPU-gyorsított, dinamikus gesztusfelismerő keretrendszer, amely olcsó RGB webkameráról fut, több analóg paramétert vezérel egyidejűleg, és iteratív fejlesztéssel fokozatosan bővített funkcionalitással rendelkezik.

---

## 3. Fejlesztési folyamat és saját eredmények

### 3.1 Hardver- és szoftverkörnyezet felállítása

A fejlesztés és kísérletezés teljes folyamata egyetlen laptopon zajlott az alábbi hardverkonfigurációval:

- **GPU:** NVIDIA Quadro P2000 (Mobile, Pascal architektúra, sm_61, 4 GB GDDR5)
- **OS:** Ubuntu (Questing)
- **Python:** 3.13

A kritikus kihívás a CUDA-kompatibilitás volt: a Pascal GPU-hoz (sm_61) az újabb PyTorch-verziók alapértelmezés szerint nem fordítanak kerneleket, ezért specifikus wheel-kombinációt alkalmaztam:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
ultralytics==8.4.19
opencv-python==4.13.0.92
```

A projekt virtuális Python-környezetben fut (`venv`), a teljes függőségi fa exportálva van a `requirements.txt` fájlba. A fejlesztés során a modellméretek a nano (yolov8n, ~6 MB) változattól a nagy (yolo26l, ~58 MB) és extra-large (yolo26x, ~126 MB) változatig terjednek; ez tette lehetővé a teljesítmény–pontosság kompromisszum empirikus tanulmányozását.

### 3.2 Testtartás-figyelő prototípus (00.py – 05.py)

A projekt első fázisát öt egymást követő, fokozatosan finomított prototípus alkotja. Ezek a fájlok (00.py, 01.py, 02.py, 03.py, 04.py, 05.py) a klasszikus szoftverfejlesztési iteratív megközelítést demonstrálják.

**00.py – alap testtartás-detektálás**

Az első prototípus betölti a YOLOv8 nano pose-modellt, megnyit egy webkameráért, majd minden egyes képkockán lefuttatja az inferenciát. A detektált 17 kulcspont közül a 0. indexű (Nose/Orr) és az 5. indexű (Left Shoulder/Bal váll) normalizált `(x, y)` koordinátáit vizsgálja. Az elsőéves hipotézis: ha az orr `y`-koordinátája megközelíti a váll `y`-koordinátáját (vagyis a kettő hányadosa egy küszöbérték alá esik), a felhasználó görnyedt. Az állapot egy zöld ("Good Posture") vagy piros ("SIT STRAIGHT!") szövegként jelenik meg a képen.

Ennél a megközelítésnél normalizált (`xyn`) koordinátákat alkalmaztam (0.0–1.0 tartomány), ami kamera-felbontás-független megítélést tesz lehetővé. A kritikus tanulság: a puszta koordináta-összehasonlítás nem elég robusztus, mert az ülőtávolság és a kameraszög változása hamis pozitív riasztásokat okoz.

**01.py – debounce-timer bevezetése**

A második iterációban pixelalapú koordinátákra tértem vissza, és bevezettem egy `time.time()`-alapú időzítőt: csak akkor riaszt a rendszer, ha a helytelen testtartás legalább 3 másodpercig fennáll (`alert_threshold = 3`). Ezzel kiszűrtem az átmeneti, rövid tartamú pozícióváltozásokat. Emellett megjelent a konfidenciaszűrés (`kpts[i][2] > 0.5`): csak akkor értékelem a kulcspontot, ha a modell legalább 50%-os valószínűséggel látja azt.

**02.py – ablak-bezárás kezelése**

A harmadik verzió kizárólag szoftverminőségi javítást hozott: az OpenCV `WND_PROP_VISIBLE` property-vel figyeli, hogy a felhasználó rákattintott-e az ablak bezáró gombjára, és erre is cleanly kilép. Ez az ipari szintű szoftverrobusztusság felé tett első lépés.

**03.py és 04.py – Nose-to-Shoulder gap kalibráció**

A negyedik és ötödik verzióban az orr-váll különbség (gap = `shoulder_y - nose_y`) abszolút pixel-értékét jelzem ki a képernyőn, lehetővé téve a felhasználó számára, hogy valós időben kalibrálja a saját `threshold` értékét. A `04.py` beállítja a küszöböt 120 pixelre, és a hibaüzenet szövegét magyarra fordítja ("Ülj egyenesen! Görnyedsz: X másodperce"), demonstrálva a lokalizáció iránti igényt.

**05.py – stabil baseline**

Az ötödik fájl a 04.py stabil másolata a finomított paraméterekkel; ez válik az egész projekt baseline-jává, amelyből a gesztusfelismerés növekszik ki.

A testtartás-detektáló fázis fő műszaki eredménye: megértettem, hogy egyetlen kulcspont-koordináta-pár nem elegendő egy robusztus testtartás-becslőhöz. Szükség van normalizálásra (skálainvariancia), időbeli szűrésre (debounce) és megbízhatósági szűrésre (konfidencia-küszöb).

### 3.3 Normalizált távolságszámítás és gesztusalap (normalize.py)

A `normalize.py` egy kísérletező szkript, amelyet a "Gesture Lab" előkészítéseként írtam. Két kézcsukló (kpt 9 és kpt 10) euklideszi távolságát a vállak közötti távolsággal (kpt 5 – kpt 6) normalizálom:

$$
d_{\text{norm}} = \frac{\| \vec{p}_{wrist\_L} - \vec{p}_{wrist\_R} \|}{\| \vec{p}_{shoulder\_L} - \vec{p}_{shoulder\_R} \|}
$$

Ez a vállszélesség-egységben kifejezett normalizált csuklótávolság skálainvariáns: értéke független attól, hogy a személy 0,5 méterre vagy 2 méterre ül a kamerától. Ez a matematikai alap az összes következő gesztusdetektor magja.

A szkript valós időben kijelzi a normalizált értéket és egy egyszerű döntési szabályt: `d_norm > 2.0` → "ARMS APART", `d_norm < 0.5` → "ARMS CLOSE". Ez volt az első PoC (proof of concept) arra, hogy a vállszélesség-normalizáció valóban skálainvariáns viselkedést eredményez.

### 3.4 Temporális gesztusfelismerés – Gesture Lab v1 (06_gestures.py)

A `06_gestures.py` két önálló gesztust valósít meg egyidejűleg, bevezetve a temporális csúszó ablak (rolling window) koncepcióját.

**Első gesztus: Karok szétnyitása/összezárása**

A normalizált csuklótávolság értékeit egy 15 elemű `deque` (kör-puffer) tárolja. A gesztus detektálásához a puffer első és utolsó értéke közötti deltát vizsgálom:

$$
\Delta = d_{\text{norm}}(t) - d_{\text{norm}}(t - 15)
$$

Ha `Δ > 0.25`, a rendszer "TÁVOLODIK", ha `Δ < -0.25`, "KÖZELEDIK" feliratot jelenít meg. A küszöbértékeket empirikus kísérletezéssel határoztam meg.

**Második gesztus: "Karmester" gesztus**

Ez az első aszimmetrikus gesztus: ha az egyik csukló az orr fölé emelkedik (master kéz), a másik csukló sebességvektora (slave kéz) meghatározza az irányt (JOBBRA, BALRA, FEL, LE). A sebesség számítása:

$$
v_x = \frac{x_{wrist}(t) - x_{wrist}(t - 15)}{d_{shoulder}}
$$

$$
v_y = \frac{y_{wrist}(t) - y_{wrist}(t - 15)}{d_{shoulder}}
$$

Ha `|vx| >= |vy|`, vízszintes irányú a mozgás; ellenkező esetben függőleges. A 0.25-ös normalizált sebességküszöb elválasztja a szándékos mozgást a kéz természetes remegésétől (jitter).

### 3.5 Összetett gesztusvezérlő rendszer – 1. iteráció (06a.py)

A `06a.py` a `06_gestures.py` teljes újraírása refaktorált, konfigurációból vezérelt architektúrával. Bevezeti a konkrét felhasználói interfész-elemeket:

**Zoom-paraméter vezérlése szétnyitás/összezárás gesztussal**

A szétnyitás-gesztus detektálásából egy `zoom_value` (0.0–1.0) skáláris értéket számolok, amelyet egy vertikális HUD-csúszó (slider) jelenít meg a képernyő jobb szélén. A tényleges zoom-faktor lineárisan interpolált: `actual_zoom = ZOOM_MIN + zoom_value * (ZOOM_MAX - ZOOM_MIN)`.

A zoom-effektus képi implementációja egy centre-crop + upscale technikával valósul meg:

```python
def apply_zoom(frame, zoom_factor):
    h, w   = frame.shape[:2]
    scale  = 1.0 / zoom_factor
    nh, nw = int(h * scale), int(w * scale)
    y1, x1 = (h - nh) // 2, (w - nw) // 2
    return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h))
```

Ez az optikai "zoom be" és "zoom ki" érzést kelti anélkül, hogy a kameraoptika változna.

**"Karmester" gesztus két tengelyen**

A karmester-gesztus outputja már nem csupán szövegcímke: a `h_value` (horizontális) és `v_value` (vertikális) csúszókat folyamatosan mozgatja a slave kéz sebességvektora alapján. Ez az első rendszer, amelyben a gesztusok analóg, folyamatos értéket vezérelnek.

A majoritásos szavazólogika (`MAJORITY_FRAC = 0.60`) és a minimális keret-delta (`MIN_FRAME_DELTA = 0.008`) szűrők kombináltankezelik a zajt: a detektáláshoz a csúszó ablak keretének legalább 60%-ában azonos irányú, elégséges amplitúdójú mozgásnak kell jelen lennie.

### 3.6 Gesztusvezérlő rendszer – 2. iteráció: zoom és fényerő (07_gesture_control.py)

A `07_gesture_control.py` lecseréli az absztrakt karmester-gesztust egy új, intuitívabb fényerő-vezérlési logikával:

**Fényerő-gesztus**

Ha mindkét csukló a váll szintje fölé emelkedik, aktiválódik a fényerő-vezérlés. A csukló-átlagmagasság és a szemek szintjének összehasonlítása határozza meg az irányt:
- Csukló átlaga a szemek felett → fényerő növelése
- Csukló a szemek és vállak közé → fényerő csökkentése

A fényerő-implementáció per-pixel float szorzóval dolgozik:

```python
def apply_brightness(frame, factor):
    scale = factor * 2.0  # 0→0, 0.5→1.0, 1.0→2.0
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)
```

**Távolságbecslés**

Bevezettem egy `dist_value` paramétert: a vállak közötti pixeltávolság és a képkeret szélességének aránya (`shoulder_dist / frame_width`) ad egy relatív közelségi mérőszámot. Ez a távolság-értéket gördülő átlaggal simítom (`DIST_SMOOTH = 10` keret), hogy zajmentes visszajelzést adjak.

**Reset-gesztus**

Ha mindkét csukló a csípőszint alá esik, a rendszer visszaállítja az összes paramétert alapértékre és törli a puffert. Ez az első "meta-gesztus": egy gesztus, amelynek nincs közvetlen vizuális hatása, csupán a rendszer belső állapotát módosítja.

### 3.7 Átállás YOLO26-ra és inverz-toggle gesztus (08yolo26.py)

A `08yolo26.py` három jelentős újítást hozott:

**YOLO26 modellcsalád integrációja**

Átálltam az Ultralytics YOLO v2.6 generációs modelljeire (`yolo26n/s/m/l/x-pose.pt`). A projekt öt különböző modellméretet tartalmaz (nano: ~7 MB, small: ~24 MB, medium: ~50 MB, large: ~58 MB, extra-large: ~126 MB). A YOLO26 architektúra jobb kulcspont-pontosságot nyújt kis méretű és részlegesen takart testek esetén, különösen a PSA (Position-aware Self-Attention) modulok révén. A modell a `results[0].plot(boxes=False)` hívással csak a kulcspont-vázat rajzolja ki, bounding box nélkül, ami tisztább vizuális megjelenítést eredményez.

**Inverz-toggle gesztus (térdemelés)**

Bevezetettem egy edge-triggered (él-érzékelő) logikát: ha bármelyik térd (kpt 13 vagy kpt 14) a csípőszint (`hip_y`) fölé emelkedik, a `invert_on` flag értéke megfordul (`cv2.bitwise_not`). Az él-érzékelés megakadályozza a folyamatos toggle-t:

```python
if knee_up and not knee_was_up:
    invert_on = not invert_on
knee_was_up = knee_up
```

**Megfordulás-kilépés**

Ha a jobb váll `x`-koordinátája nagyobb lesz, mint a bal vállé (vagyis a személy háttal fordul a kamerának), a program kilép. Ez egy intuitív, természetes "kilépés gesztus": megfordulok → vége.

```python
if rs[0] > ls[0]:
    break
```

Ez az ellenőrzés kizárólag a YOLO26-nál van implementálva (a mirroring-mentes konfigurációban), ahol a modell a valódi bal/jobb oldalakat érzékeli.

**Display scale**

Bevezetettem egy `DISPLAY_SCALE = 2` paramétert, amely a kamera 1280×720-as képét 2-szeres méretűre nagyítja a megjelenítéskor (`cv2.INTER_NEAREST`). Ez az alacsony kamera-felbontás melletti jobb láthatóságot biztosít a fejlesztés során.

### 3.8 Finomított zoom-logika és ablakméretezés (09zoom.py)

A `09zoom.py` a zoom-logikát finomítja két aspektusból:

**Zoom csak vízszintes kéztartásnál**

Bevezetettem az `ALIGN_THRESH = 0.5` paramétert: a zoom-gesztus csak akkor aktív, ha a két csukló függőleges eltérése (`norm_vdiff`) a vállszélesség 50%-án belül van. Ez megakadályozza, hogy a fényerő-gesztus (egyik kéz fent, másik lent) véletlenül zoom-ot indítson el.

Ezzel a két gesztus – zoom és fényerő – egymást kizáró, egyidejűleg nem aktiválható, ami teljesen eliminálja az interferencia-problémát.

**Zoom-alapállapot és reset**

A `zoom_value` alapértéke `1/3`, ami pontosan 1.0×-es zoom-faktort jelent (`ZOOM_MIN=0.5, ZOOM_MAX=2.0`): `0.5 + 1/3 * 1.5 = 1.0`. Ez az intuitív alapállapot, ahol a kép nem torzul.

**Dinamikus ablakméretezés**

A megjelenítő ablak méretét maga a zoom-faktor határozza meg – valódi kinagyított ablakot kapunk, nem csak a kép egy részét kitöltő hatást:

```python
if actual_zoom != 1.0:
    canvas = cv2.resize(canvas, (int(fw * actual_zoom), int(fh * actual_zoom)),
                        interpolation=cv2.INTER_LINEAR)
```

### 3.9 RGB-csatorna-vezérlés és álló-detektálás (10rgb.py, 11detect_standing.py)

#### 10rgb.py – RGB-csatorna-vezérlés

A `10rgb.py` kibővíti a fényerő-vezérlést két egyedi szín-csatorna vezérlésével:

| Gesztus | Hatás |
|---------|-------|
| Mindkét kéz váll felett | Fényerő (mindhárom csatorna) |
| Csak bal kéz váll felett | Vörös (R) csatorna |
| Csak jobb kéz váll felett | Zöld (G) csatorna |
| Mindkét kéz csípő alatt | Reset |
| Térdemelés | Inverz toggle |
| Megfordulás | Kilépés |

Az RGB-alkalmazás per-csatorna float szorzással valósul meg:

```python
def apply_color(frame, bright, red, green):
    f = frame.astype(np.float32)
    out[:, :, 0] = np.clip(f[:, :, 0] * b,      0, 255)  # B
    out[:, :, 1] = np.clip(f[:, :, 1] * b * g,  0, 255)  # G
    out[:, :, 2] = np.clip(f[:, :, 2] * b * r,  0, 255)  # R
    return out.astype(np.uint8)
```

Ez 5 párhuzamos HUD-csúszót jelenít meg (ZOOM, BRT, RED, GRN + aktuális zoom-érték), teljes körű vizuális visszajelzéssel.

#### 11detect_standing.py – álló személy detektálása

A legfejlettebb verzió (11detect_standing.py) bevezet egy újabb biztonsági réteget: a gesztúrák csak akkor aktiválódnak, ha a rendszer álló személyt detektál.

Az álló-detektálás feltételrendszere:
1. A bounding box és a keypoint-adatok elérhetők
2. A 0–12. kulcspontok (orrtól csípőig) mindegyike konfidenciaküszöb felett van (`> 0.5`)
3. Minden látható kulcspont a bounding boxon belül van

```python
all_visible = all(kp[i][2].item() > 0.5 for i in range(13))
all_inside  = all(
    x1 <= kp[i][0].item() <= x2 and y1 <= kp[i][1].item() <= y2
    for i in range(13) if kp[i][2].item() > 0.5
)
is_standing = all_visible and all_inside
```

Ezt az állapotot egy HUD-badge jelzi ("STANDING" / "NOT STANDING"). A geszturvezérlés csak `is_standing == True` esetén aktív. Ez drasztikusan csökkenti a téves aktiválások számát (pl. amikor valaki ülve hajol be a kamera elé).

### 3.10 Egérvezérlés testtartással (mouse.py)

A `mouse.py` egy teljesen eltérő alkalmazást implementál: a jobb csukló (kpt 10) helyzetének kameraközépponttól vett eltérése mozgatja az operációs rendszer egérkurzorát, a bal csukló bal szem fölé emelése bal kattintást vált ki.

**Holt-zóna (deadzone)**

Egy 50 pixel sugarú "semleges zóna" körül a kameraközéppontnál: csak azon belülről kilépve indul az egérmozgás. Ez megakadályozza a kézremegés miatti akaratlan egérmozgást.

**Exponenciális simítás (EWMA)**

Az egérmozgás simítása exponential weighted moving average (EWMA) szűrővel:

```
smooth_vx = SMOOTHING * smooth_vx + (1 - SMOOTHING) * raw_move_x
```

`SMOOTHING = 0.7` esetén a rendszer az előző keret sebességének 70%-át és az aktuális keret nyers sebességének 30%-át kombinálja. Ez természetes, "gliding" egérmozgást eredményez.

**Kattintás debounce**

Az `IS_CLICKING` flag megakadályozza, hogy egy emelt kéz folyamatos kattintás-áradatot okozzon: egyszer kattint emeléskor, majd csak a kéz leengedése és újbóli emelése esetén kattint ismét.

A `pyautogui.FAILSAFE = False` beállítás letiltja a fail-safe sarok-mechanizmust (amely alapértelmezetten a képernyő sarkába mozgatott egérnél letiltja a pyautogui-t), mivel a geszturvezérlés közben az egér természetesen elérheti a sarkokat.

### 3.11 ASL-ábécé felismerés – tanítási kísérlet (training/)

A projekt explorációs ágát egy American Sign Language (ASL) ábécéfelismerő modell tanítása képezi. A `training/` könyvtár tartalmaz:

- **asl_data.yaml**: 26 osztálydefiníció (A–Z betűk) COCO-YOLO formátumban
- **train_asl.py**: tanítási szkript YOLOv8n detektáló modellre

A tanítás konfigurációja:
```
epochs=50, imgsz=640, batch=16, device=0
```

A YOLOv8n nano modell választása tudatos: a Quadro P2000 4 GB VRAM-jára optimalizálva, a kis batch-méret (16) megakadályozza az OOM (out of memory) hibákat. Ez a kísérlet demonstrálja, hogy az Ultralytics ökoszisztéma egységes API-ja lehetővé teszi az átmenetet a pre-trained pose-modellek felhasználásától (zero-shot inference) a saját adathalmazon tanított, domain-specifikus detektorig.

---

## 4. A saját és korábbi eredmények összehasonlítása

### 4.1 Funkcionális összehasonlítás hasonló rendszerekkel

| Jellemző | Saját rendszer | MediaPipe + custom | AirTouch | GestureBot (Kinect) |
|----------|---------------|-------------------|----------|---------------------|
| Szenzor | RGB webkamera | RGB webkamera | Beépített TV-kamera | RGB-D (Kinect) |
| GPU-gyorsítás | Igen (CUDA) | Nem | Nem | Igen (Kinect SDK) |
| FPS (target) | 30–60+ | 15–30 | ~15 | 30 |
| Gesztusok száma | 7+ | 3–5 | 4–6 | 10+ |
| Analóg vezérlés | Igen | Részben | Nem | Igen |
| Álló-detektálás | Igen | Nem | Nem | Igen |
| Temporális szűrés | Igen | Részben | Nem | Igen |
| Open-source | Igen | Részben | Nem | Nem |
| Mélységszenzor szükséges | Nem | Nem | Nem | Igen |

### 4.2 Architektúrális evolúció a projekt során

A projekt 12 fő fejlesztési iteráción ment keresztül 2026. március 3. és március 17. között. Az alábbi táblázat összefoglalja a legfontosabb műszaki hozzájárulásokat:

| Verzió | Fő újítás | Műszaki tudás |
|--------|-----------|---------------|
| 00.py | Alapdetektálás | YOLOv8 pose API |
| 01.py | Debounce-timer | Időbeli szűrés |
| 02.py | Ablakkezelés | Robusztus UI |
| 03–04.py | Gap kijelzés, lokalizáció | Kalibráció és i18n |
| normalize.py | Vállszélesség-normalizáció | Skálainvariancia |
| 06_gestures.py | Rolling window, karmester | Temporális gesztusok |
| 06a.py | Majoritásos döntés, slider UI | Zajszűrés, analóg vezérlés |
| 07.py | Fényerő-zóna, távolságbecslés | Zóna-hierarchia |
| 08yolo26.py | YOLO26, invert-toggle, megfordulás-kilépés | Modellmigráció, meta-gesztura |
| 09zoom.py | Vízszintes igazítás, dinamikus ablak | Gesztúra-exkluzivitás |
| 10rgb.py | RGB-csatorna vezérlés | Per-csatorna képfeldolgozás |
| 11detect_standing.py | Álló-detektálás | Kulcspont-láthatóság feltétel |
| mouse.py | Egérvezérlés, EWMA simítás | HCI alkalmazás |

### 4.3 YOLOv8 vs. YOLO26 – saját mérések

A 08yolo26.py bevezetésekor megfigyelt empirikus különbségek:

**Pontosság:** A YOLO26 nano variáns noticeably robusztusabban detektálja a kulcspontokat félprofilból és részleges takarás esetén, ami kritikus a "megfordulás-kilépés" gesztúra megbízhatóságához.

**Inferencia sebesség:** A Quadro P2000-en mindkét nano variáns ~30–40 FPS-t nyújt. Az s (small) variáns ~20–25 FPS-re csökkenti a sebességet, de javítja a pontosságot. Az l és x variánsok ~10–15 FPS-en futnak, ami már nem elégséges valós idejű interakcióhoz ezen a hardveren.

**Modellméret:** yolo26n (~7 MB) vs. yolov8n (~6 MB) – elhanyagolható különbség a betöltési időben.

A saját mérések azt mutatják, hogy a YOLO26n–YOLO26s tartomány az optimális választás a Quadro P2000 hardverrel valós idejű geszturvezérléshez.

---

## 5. Továbbfejlesztési lehetőségek

### 5.1 Temporális modellezés gépi tanulással

A jelenlegi gesztusfelismerés szabályalapú: küszöbértékek és majoritásos szavazás. Jövőbeli irány a gépi tanulásra épülő temporális osztályozó bevezetése, például:
- **LSTM (Long Short-Term Memory):** A csúszó ablak keypoint-sorozatait osztályozza diszkrét gesztúrákba. Alkalmas jeltolmácsolásra.
- **Transformer (attention-based):** Jobb hosszú-távú függőségek kezelésére; különösen dinamikus kézmozdulatok (írás levegőbe, ASL szavak) esetén.

### 5.2 3D mélység-integráció

A jelenlegi rendszer egyetlen RGB kamerával 2D kulcspontokat becsül. A mélységinformáció (depth) integrálása – akár sztereokamerával, akár Intel RealSense szenzorral – lehetővé tenné:
- Valódi 3D kulcspont-koordináták számítását
- Robusztusabb occlusion-kezelést
- Természetesebb, 3D térben értelmezett gesztusokat (pl. "push" vagy "pull")

### 5.3 Többszemélyes geszturvezérlés

A jelenlegi implementáció kizárólag az első detektált személyt kezeli (`keypoints.data[0]`). Az Ultralytics framework keretein belül a `keypoints.data` több személyhez is tartalmaz adatokat; a kiterjesztés lehetővé tenné több személy egyidejű, különböző paramétereket vezérlő szerepét.

### 5.4 ASL-felismerés integrálása valós idejű rendszerbe

A tanítási kísérlet (`training/`) eredményeként kapott detektor-modell beilleszthető a geszturvezérlő pipeline-ba: a kézgesztusok és a testpóz-gesztusok kombinációja gazdag bemeneti csatornát biztosítana (pl. ASL-betűk + testpóz = komplex parancsnyelv).

### 5.5 Kalibrációs modul

Felhasználó-specifikus küszöbértékek automatikus kalibrálása: a rendszer az első 30 másodpercben méri a felhasználó természetes mozgástartományát és statisztikusan állítja be a küszöbértékeket. Ez megküntetné a manuális paraméterbeállítás szükségességét.

### 5.6 Alkalmazásintegráció

A jelenlegi rendszer a kép-paramétereket (zoom, fényerő, RGB) csak a saját ablakán belül manipulálja. Jövőbeli irány:
- OS-szintű médialejátszó-vezérlés (hangerő, következő dal) DBus-on keresztül
- Prezentáció-vezérlés (dia előre/hátra) xdotool-lal
- Böngésző-navigáció (scroll, tab-váltás) xdg-open-nal

---

## 6. Összegzés

A projekt során egy iteratív fejlesztési folyamatban egy valós idejű, kamera-alapú geszturvezérlő rendszert valósítottam meg YOLOv8 és YOLO26 testpóz-becsléssel, Python/OpenCV keretrendszerrel, NVIDIA CUDA GPU gyorsítással. Az iterációk száma 12, a kódsorok száma ~1800+, a fejlesztési időszak 14 nap.

**Saját eredményeim:**

1. Megvalósítottam és finomítottam egy vállszélesség-alapú normalizált gesztusfelismerőt, amely skálainvariáns viselkedést biztosít különböző kamera-távolságok esetén.

2. Kidolgoztam egy többrétegű zajszűrési stratégiát: konfidencia-küszöb, debounce-timer, temporális rolling window, majoritásos szavazás és gesztúra-exkluzivitás. Ezek kombinációja robusztus, valós körülmények között (változó megvilágítás, mozgó háttér) megbízható felismerést nyújt.

3. Implementáltam egy testpóz-zóna-hierarchiát (csípő alatti / váll-csípő közötti / váll–szem közötti / szem feletti zóna), amely szemantikailag értelmes, intuitív gesztursteret hoz létre.

4. Elvégeztem egy YOLO26 vs. YOLOv8 összehasonlítást valós idejű geszturvezérlő alkalmazásban, és meghatároztam az optimális modell-méretet (yolo26n/s) a Quadro P2000 hardverkörnyezetre.

5. Megvalósítottam az álló személy detektálásán alapuló gesztúra-aktiválást, amely megakadályozza a téves aktiválásokat nem szándékos testpozíciókból.

6. Elkészítettem egy egérvezérlő modult EWMA-simítással és debounce-kattintás-logikával, demonstrálva a rendszer HCI alkalmazhatóságát.

7. Elvégeztem egy ASL-ábécé felismerő modell tanítási kísérletét, demonstrálva a zero-shot → fine-tuned pipeline átmenetet az Ultralytics ökoszisztémán belül.

A fejlesztés során szerzett tapasztalat megerősítette: a valós idejű geszturvezérlés kulcsa nem a modell bonyolultsága, hanem a robusztus normalizáció, a megfelelő temporális szűrés és az intuatív gesztura-tér megtervezése.

---

## 7. Irodalomjegyzék

1. **Redmon, J. et al. (2016)**: You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the IEEE CVPR*, pp. 779–788.

2. **Jocher, G. et al. (2023)**: Ultralytics YOLOv8. GitHub repository. Ultralytics. https://github.com/ultralytics/ultralytics

3. **Cao, Z. et al. (2017)**: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. *Proceedings of the IEEE CVPR*, pp. 7291–7299.

4. **Lugaresi, C. et al. (2019)**: MediaPipe: A Framework for Perceiving and Processing Reality. *Third Workshop on Computer Vision for AR/VR at IEEE CVPR*.

5. **Lin, T.-Y. et al. (2014)**: Microsoft COCO: Common Objects in Context. *European Conference on Computer Vision (ECCV)*, pp. 740–755.

6. **Zhang, F. et al. (2020)**: MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv preprint arXiv:2006.10214*.

7. **Maji, D. et al. (2022)**: YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss. *arXiv preprint arXiv:2204.06806*.

8. **Bradski, G. (2000)**: The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 25(11), pp. 120–125.

9. **Paszke, A. et al. (2019)**: PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, pp. 8024–8035.

10. **Glorot, X. & Bengio, Y. (2010)**: Understanding the Difficulty of Training Deep Feedforward Neural Networks. *Proceedings of the 13th AISTATS*, pp. 249–256.

---

*A jelen írásbeli beszámoló a VEMIINM336L tárgy követelményei szerint készült. Az elkészítés során mesterséges intelligencia eszközök (Claude Sonnet) alkalmazása megengedett a tárgy tematikája szerint.*

*Összes kód elérhető a fejlesztési mappában: `/home/rama/Desktop/yolo/`*
