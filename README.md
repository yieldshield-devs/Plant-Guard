# Smart Farming AI 🌱

An IoT-enabled smart agriculture system built on **ESP32-S3**. It brings together
a camera node, soil and environmental sensors, an on-device **TensorFlow Lite
Micro** plant-disease classifier, and a **4G modem that sends bilingual
(English + Bengali) SMS alerts** to farmers.

The system is designed for eggplant (brinjal) cultivation and performs disease
inference entirely at the edge — no cloud connection is required for a
prediction.

## Overview

There are two firmware roles:

- **Main Node ("God Node")** — an ESP32-S3 that hosts a Wi-Fi access point and
  web server, runs the eggplant disease classifier, reads all soil/environment
  sensors, and sends an SMS report through an A7670C 4G modem.
- **Camera Node** — a remote ESP32-S3 with an OV3660 camera and a moisture
  sensor. It captures a JPEG, reads soil moisture, and uploads both to the Main
  Node over Wi-Fi.

When the Camera Node delivers an image, the Main Node preprocesses it, runs the
TFLite model, reads the sensors, and sends one combined SMS report per detection.

## Key Features

- **Edge AI:** a 7-class eggplant leaf classifier runs on the ESP32-S3 using
  TensorFlow Lite Micro.
- **JPEG + quantization aware:** images are decoded with JPEGDEC, center-cropped,
  and written directly into the model input tensor with the correct
  quantization (supports `uint8` and `int8` input types).
- **Disease detection:** predicts one of seven conditions from a plant image.
- **Bilingual advice:** English and Bengali guidance per class, loaded from a
  JSON file on an SD card.
- **Soil & environment monitoring:** NPK (RS485 Modbus), pH, DHT22 temperature /
  humidity, and an ultrasonic stagnant-water-level sensor.
- **SMS alerting:** sends the detection result and full sensor report to two
  phone numbers via an A7670C modem (UCS2 encoding for Bengali text).
- **Remote camera node:** a second board pushes photos and moisture over Wi-Fi.
- **Web UI:** the Main Node serves a simple page for manual image upload and
  live sensor values.

## System Architecture

```text
  Camera Node (ESP32-S3 N16R8 + OV3660 + moisture)
        │
        │  Wi-Fi (STA → Main Node AP "SmartFarm-S3")
        │  GET  /moisture?node=1&value=67
        │  POST /camera   (raw JPEG)
        ▼
  Main Node "God Node" (ESP32-S3, 7 MB tensor arena in PSRAM)
        ├── TensorFlow Lite Micro  → Eggplant 7-class classifier
        ├── NPK sensor (RS485 / Modbus RTU)
        ├── pH sensor (analog)
        ├── DHT22 (temperature + humidity)
        ├── Ultrasonic (stagnant water level)
        ├── SD card → plant_guard_bangla_sms.json (bilingual advice)
        ├── Wi-Fi SoftAP + WebServer (manual upload page)
        └── A7670C 4G modem → bilingual SMS to two numbers
```

Processing flow:

1. **Capture:** the Camera Node captures a JPEG (VGA) with the OV3660.
2. **Upload:** it sends moisture to `/moisture`, then the JPEG to `/camera`.
3. **Preprocess:** the Main Node center-crops and scales the JPEG into the model
   input tensor.
4. **Inference:** TensorFlow Lite Micro classifies the leaf.
5. **Sensor read:** NPK, pH, DHT22, and water level are sampled.
6. **Report:** a bilingual SMS with the diagnosis, advice, and sensor values is
   sent to both configured numbers.

> **Important:** Model accuracy depends on the training dataset, image quality,
> lighting, camera distance, and how closely field images match the training
> data. Validate predictions against real plant samples before using them to
> make crop-treatment decisions.

## Machine Learning Model

The classifier is a **7-class eggplant leaf-disease image classifier**, trained
in **Edge Impulse** (project `eggplantv2`, project ID `1116746`) as a
**MobileNetV2 transfer-learning** model and exported to TensorFlow Lite.

Verified tensor layout (parsed from the exported model):

| | Shape | Type | Scale | Zero point |
|---|---|---|---|---|
| Input  | `[1, 160, 160, 3]` | INT8 | `1/255`  | `-128` |
| Output | `[1, 7]`            | INT8 | `1/256`  | `-128` |

The graph uses standard TFLite Micro operators only — `CONV_2D`,
`DEPTHWISE_CONV_2D`, `ADD`, `FULLY_CONNECTED`, `RESHAPE`, and `SOFTMAX` — so it
runs with the stock `AllOpsResolver`.

### Model files (`models/`)

| File | Notes |
|---|---|
| `ei-eggplantv2-transfer-learning-tensorflow-lite-int8-quantized-model.6.lite` | **Use this one.** Standard TFLite, INT8. |
| `ei-eggplantv2-transfer-learning-tensorflow-lite-float32-model.6.lite` | Standard TFLite, float32 (not used by the current firmware). |
| `eggplantv2-cpp-mcu-v{1,2}-impulse-#1.zip` | Edge Impulse **EON-compiled** C++ SDK exports (require the Edge Impulse SDK). |
| `ei-eggplantv2-arduino-1.0.3-impulse-#1.zip` | Edge Impulse Arduino library (EON-compiled). |

### Embedding the model

The firmware runs the model through its own TFLite Micro interpreter and expects
the flatbuffer as a C array:

```cpp
extern const unsigned char g_model[];
extern const unsigned int g_model_len;
```

Generate `model_data.cc` / `model_data.h` from the **int8** `.lite` file (for
example with `xxd -i`) and add it to the sketch. The Edge Impulse `.zip` exports
do **not** provide `g_model`; they expose an EON runtime API
(`tflite_learn_1116746_6_init/_invoke/_input/_output`) that would instead require
including the Edge Impulse SDK in place of the stock TFLite Micro library.

> **Input quantization requirement:** this model expects **normalized** input in
> `[0, 1]` (input scale `1/255`, zero point `-128`). A raw pixel `p` (0–255)
> must therefore be quantized as
> `q = round((p / 255.0) / scale) + zero_point`, which simplifies to `p - 128`.
> Quantizing as `p / scale` saturates every non-zero pixel and yields invalid
> predictions.

### Datasets

The model was trained / fine-tuned using these public leaf-disease image
datasets:

- **PlantVillage** — a large, well-labeled collection of plant leaf images
  (healthy and diseased) captured under controlled, mostly uniform backgrounds.
  It is a common baseline for leaf-disease classification.
- **PlantDoc** — a dataset of **real-world field** leaf images with varied
  backgrounds, lighting, and occlusion, used to make the model more robust to
  in-field conditions that differ from PlantVillage.

Because of the domain gap between the two datasets, validate the model on field
images before deployment.

### Known issues

- The `g_model` byte array is **not yet embedded**, so the Main Node sketch will
  not link until `model_data.cc` is generated from the int8 `.lite` and added.
- The int8 input preprocessing in `drawJPEG()` (`prototype5.ino`) currently
  quantizes as `round(pixel / scale) + zero_point`, which is incorrect for this
  model's `1/255` input scale and saturates the input. It must normalize to
  `[0, 1]` first.

## Disease Classes

The classifier outputs the following 7 classes, in this order:

| # | Firmware class | Edge Impulse label | Bengali status |
|---|----------------|--------------------|----------------|
| 0 | Healthy Leaf   | `healthy_leaf`     | সুস্থ পাতা     |
| 1 | Insect Pest    | `insect_pest`      | পোকামাকড়     |
| 2 | Leaf Spot      | `leaf_spot`        | পাতায় দাগ    |
| 3 | Mosaic Virus   | `mosaic_virus`     | মোজাইক রোগ    |
| 4 | Small Leaf     | `small_leaf`       | ছোট পাতা     |
| 5 | White Mold     | `white_mold`       | সাদা ছত্রাক    |
| 6 | Wilt Disease   | `wilt_leaf`        | ঢলে পড়া রোগ  |

The class **order** must match the order used during training. The firmware and
the Edge Impulse export use slightly different display names (for example
`Wilt Disease` vs. `wilt_leaf`), but the index order is identical.

## Repository Structure

```text
Codes/
├── firmware/
│   ├── prototype5/
│   │   └── prototype5.ino   # Main Node ("God Node")
│   └── CamNode/
│       └── CamNode.ino      # Camera + moisture node
├── models/                  # TFLite models + Edge Impulse exports
├── LICENSE
└── README.md
```

> **Model file:** the Main Node expects the model as `g_model[]` / `g_model_len`
> (declared `extern` at the top of `prototype5.ino`). A `model_data.cc` /
> `model_data.h` generated with `xxd -i` from the int8 `.lite` file must be added
> to the build. Large binaries are best kept out of normal Git history (Git LFS
> or release assets).

## Hardware

**Main Node**

| Component              | Interface / notes                       |
|------------------------|-----------------------------------------|
| ESP32-S3 board (N16R8) | PSRAM **required** (7 MB tensor arena)  |
| A7670C 4G modem        | UART, powers on via PWRKEY              |
| NPK soil sensor        | RS485 (MAX485-style transceiver)        |
| pH sensor              | Analog output + T1 pin                  |
| DHT22                  | Temperature + humidity                  |
| Ultrasonic (HC-SR04)   | Water level                             |
| microSD card reader    | SPI, holds `plant_guard_bangla_sms.json`|

**Camera Node**

| Component              | Interface / notes                       |
|------------------------|-----------------------------------------|
| ESP32-S3 N16R8         | PSRAM used for camera framebuffers      |
| OV3660 camera          | 8-bit parallel camera bus               |
| Capacitive moisture    | Analog output                           |

### Main Node pinout

```text
A7670C        GPIO17 <- TX      GPIO18 -> RX      GPIO16 -> PWRKEY
NPK RS485     GPIO8  -> DI      GPIO12 -> DE      GPIO11 -> RE      GPIO3 <- RO
DHT22         GPIO4  <- DATA
pH sensor     GPIO2  <- PO      GPIO42 <- T1
Ultrasonic    GPIO48 -> TRIG    GPIO47 <- ECHO
SD card       GPIO37 -> CS      GPIO38 -> SCK     GPIO39 -> MOSI    GPIO40 <- MISO
```

### Camera Node pinout (OV3660)

```text
XCLK 15   SIOD 4   SIOC 5    PCLK 13   VSYNC 6   HREF 7
Y2 11  Y3 9  Y4 8  Y5 10  Y6 12  Y7 18  Y8 17  Y9 16
Moisture (analog) GPIO1
```

Use appropriate level shifting and power conditioning where necessary, and
verify that camera/GPIO pins do not conflict with PSRAM or flash on your
specific ESP32-S3 module.

## Configuration

Values you will likely need to change before deployment:

**Camera Node (`firmware/CamNode/CamNode.ino`)**

- `SSID` / `PASSWORD` — Wi-Fi network of the Main Node.
- `CAMERA_SERVER` / `MOISTURE_SERVER` — Main Node endpoints.
- `NODE_ID`, `MOISTURE_PIN`, `DRY_VALUE`, `WET_VALUE` — moisture calibration.

**Main Node (`firmware/prototype5/prototype5.ino`)**

- `SSID` / `PASS` — SoftAP credentials (Camera Node must match).
- `PHONE_1` / `PHONE_2` — SMS recipients (in international format).
- `PH7_VOLTAGE` / `PH_SLOPE` — pH calibration, **must** be calibrated with known
  buffer solutions; they are not universal constants.
- `WATER_EMPTY_DISTANCE_INCH` — distance from the ultrasonic sensor to the empty
  container bottom.
- `SOIL_MOISTURE_ALERT_PERCENT` — threshold below which the "water the plant"
  line is added.
- `ADVICE_JSON_PATH` — SD path of the advice file (`/plant_guard_bangla_sms.json`).

### Advice JSON format

The SD card file is a flat JSON object keyed by the Bengali class names used in
`ADVICE_KEYS`:

```json
{
  "পোকামাকড়": "…",
  "পাতায় দাগ": "…",
  "মোজাইক ভাইরাস": "…",
  "ছোট পাতা": "…",
  "সাদা ছত্রাক": "…",
  "ঢলে পড়া রোগ": "…"
}
```

If the file is missing or unreadable, the firmware falls back to built-in
English/Bengali advice strings.

## Interfaces

### HTTP API (Main Node, SoftAP `192.168.4.1`)

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| GET  | `/` | Web UI: manual upload form + current sensor values | HTML |
| POST | `/upload` | Multipart upload (field `photo`, JPEG) → runs the classifier | `RESULT: <class>` |
| POST | `/camera` | Raw JPEG body from a Camera Node → runs the classifier | `CAMERA RESULT: <class>` |
| GET  | `/moisture?node=<n>&value=<pct>` | Camera Node pushes soil moisture (0–100 %) | `MOISTURE RECEIVED` |

- Valid moisture nodes are `1`–`MAX_MOISTURE_NODES` (10). Only **node 1**
  moisture is included in the SMS.
- Every successful detection increments an internal image counter and sends one
  SMS report (`<n>th image received`).

### Serial (Main Node)

- Runs at **115200 baud** and prints boot diagnostics, sensor readings, ML
  results, and SMS status.
- Any line typed into the serial monitor is forwarded to the A7670C modem as a
  raw AT command (manual modem control).

### Camera Node behaviour

- The Camera Node performs **one transmission per reset**: it sends moisture to
  `/moisture`, then the JPEG to `/camera`, then waits. Press RESET for another.

## SMS Report

Each detection sends one bilingual report (UCS2-encoded) to **both** configured
numbers. The message contains:

1. `PLANT GUARD ALERT / প্ল্যান্ট গার্ড সতর্কতা` header and the image number.
2. **Plant status** — English class name + Bengali status.
3. **Suggestion** (English) and **পরামর্শ** (Bengali), from the advice JSON.
4. **NPK** — nitrogen, phosphorus, potassium in ppm (or an error line).
5. **Water level** — in inches (or "not available in this crop").
6. **pH** — one decimal place (or an error line).
7. **Moisture** — camera-node soil moisture %, with a
   `Water the Plant (গাছে জল দাও)` line when it falls below
   `SOIL_MOISTURE_ALERT_PERCENT` (default 60 %).

## Camera Settings

The Camera Node captures **VGA (640×480) JPEG at quality 12** with framebuffers
in PSRAM, using the OV3660 sensor driver. The Main Node center-crops the image to
a square and scales it to the model input size (**160×160**) before inference.

## Software Requirements

- Arduino IDE or PlatformIO
- ESP32 board support package
- [JPEGDEC](https://github.com/bitbank2/JPEGDEC) — JPEG decode + tiling into the input tensor
- DHT sensor library (Adafruit) — DHT22
- ArduinoJson — advice JSON parsing
- **Chirale_TensorFlowLite** — TFLite Micro runtime used by the firmware
- `esp32-camera` — camera driver (Camera Node)
- `SD` / `SPI` (bundled with the ESP32 core) — Main Node SD card
- `xxd` (or equivalent) — to convert the `.lite` model into a C array

Record the exact library versions you test with before a final deployment.

## Getting Started

1. Clone or download this repository.
2. Open each `.ino` sketch (`firmware/prototype5` and `firmware/CamNode`) in
   Arduino IDE or PlatformIO.
3. Install the ESP32 board package and the libraries listed above.
4. Generate `model_data.cc` / `.h` from the **int8** `.lite` model and add it to
   the Main Node sketch.
5. Copy `plant_guard_bangla_sms.json` to the root of the SD card.
6. Update the configuration values in both sketches (Wi-Fi, endpoints, phone
   numbers, calibration).
7. Connect the camera and sensors according to your wiring diagram.
8. Select the correct ESP32-S3 board (PSRAM enabled) and port for each board.
9. Build and upload both sketches.
10. Open the serial monitor at **115200 baud**.
11. Reset the Camera Node to trigger one moisture + photo transmission; watch the
    Main Node log the inference result and SMS status.
12. Verify sensor readings and predictions with known samples before field use.

The Main Node web UI is reachable at its SoftAP IP (`192.168.4.1` by default);
a JPEG can also be uploaded there manually for testing.

## Testing and Validation

- Verify each sensor independently before integrating the ML model.
- Test model inference on a laptop using the same preprocessing (center-crop +
  `[0, 1]` normalization + quantization) as the firmware.
- Compare embedded predictions with laptop predictions on identical images.
- Evaluate on unseen images captured under different lighting, backgrounds, and
  distances, and compare performance on PlantVillage-style vs. PlantDoc-style
  field images.
- Report class-wise precision/recall or a confusion matrix when available.
- Measure inference time, memory use, and reliability on the target hardware.
- Confirm SMS delivery and Bengali (UCS2) rendering on the target networks.

## Limitations

- Image classification is not a substitute for expert diagnosis.
- Similar-looking symptoms, poor lighting, occlusion, and distant or
  low-resolution images can cause incorrect predictions.
- The model is trained for eggplant; do **not** assume it generalizes to other
  crops.
- Models trained largely on PlantVillage (controlled backgrounds) can lose
  accuracy on PlantDoc-style field images with cluttered backgrounds.
- Sensor readings require calibration and may vary with installation and soil
  conditions.
- Disease predictions should not automatically trigger pesticide or treatment
  decisions without validation.
- SMS delivery depends on cellular coverage and SIM/network support for UCS2.

## Future Improvements

- Add a dashboard for live sensor data and predictions.
- Log readings and predictions with timestamps.
- Improve the dataset with verified field images for the target crop.
- Add confidence thresholds and an "uncertain" result for low-confidence
  predictions.
- Optimize inference and memory use for the ESP32-S3.
- Add offline data storage and optional cloud synchronization.

## Contributing

Contributions are welcome. Document hardware changes, keep class labels and the
advice JSON aligned with the trained model, and validate changes on the target
board before deployment.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `PSRAM REQUIRED!` and halt | Board is not an N16R8 / PSRAM disabled. Enable PSRAM in the board settings. |
| `Camera initialization failed` | Wrong camera pins or sensor; verify the OV3660 pinout and PSRAM availability. |
| `SD CARD INITIALIZATION FAILED` | No/failed SD card, wrong SPI pins, or not FAT-formatted. Advice falls back to built-in strings. |
| `A7670C NOT RESPONDING` | Modem power / SIM / UART wiring; check `PWRKEY`, TX↔RX crossover, and SIM insertion. |
| `SMS FAILED` / `NETWORK NOT REGISTERED` | No cellular coverage, or the SIM is not provisioned for UCS2 SMS. |
| `TENSOR ARENA FAILED` / `ALLOCATE TENSORS FAILED` | Insufficient PSRAM — the model requires PSRAM. |
| Build error: undefined `g_model` | `model_data.cc` not generated/added — see *Embedding the model*. |
| Wrong / garbage predictions | Confirm input normalization (`[0, 1]`, i.e. `pixel - 128`); see *Known issues*. |

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE)
for the full text.

> **Third-party components:** the pretrained classifier and the Edge Impulse
> export archives in `models/*.zip` are generated by Edge Impulse and are subject
> to Edge Impulse's own license terms (an active Edge Impulse subscription is
> required to use them). The PlantVillage and PlantDoc datasets also carry their
> own respective licenses. The MIT license above covers only this repository's
> own source code.
