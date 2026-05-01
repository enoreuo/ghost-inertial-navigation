# GHOST: A GPS-Free Inertial Navigation System for Autonomous Vehicles

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **A learned 3-D displacement network tightly integrated with the ArduPilot EKF3 sensor-fusion stack.**

GHOST replaces a GPS receiver with a TensorFlow Lite Micro neural network running directly inside the ArduPilot autopilot at 200 Hz on STM32H7-class flight controllers. The network outputs 3-D body-frame displacement plus per-axis uncertainty, and the C++ `AP_GHOST` library publishes these as `VISION_POSITION_ESTIMATE` messages with adaptive covariance to EKF3 — making GHOST a drop-in GPS replacement that works in tunnels, urban canyons, indoor environments, and other GNSS-denied scenarios.

---

## Highlights

- **0.148 m mean 3-D error** on the public `tlio_golden` benchmark (67,323 windows).
- **Real-time on a microcontroller** — TFLM-deployable v2.5 architecture running in ~10–20 ms per inference on a CubeOrangePlus (STM32H7 @ 480 MHz).
- **Tight EKF3 integration** — uses ArduPilot's standard `ExternalNav` source (`EK3_SRC1_POSXY = 6`); the rest of the autopilot stack is unchanged.
- **Calibrated uncertainty** — heteroscedastic NLL training produces per-axis log-variance the EKF consumes directly.
- **Boot-validated on real hardware** — firmware builds and runs on a real CubeOrangePlus; EKF3 confirms `using external nav data`.

---

## What's in this repository

```
.
├── lab_tensor.ipynb           # Training notebook (produces GHOST_v25_TFLM.keras)
├── lab_car_sim.ipynb          # Evaluation notebook (regenerates paper figures)
├── lab_Doc.ipynb              # Comprehensive math documentation
├── model_arch_v25_tflm.py     # v2.5 architecture: multi-scale + dilated + SE
├── load_data_tf.py            # tlio_golden / EuRoC / TUM-VI data loader
├── train_tf.py                # NLL loss, displacement MAE metric
├── GHOST_v25_TFLM.keras       # Pre-trained weights (~7 MB)
├── GHOST_v25_TFLM.png         # Architecture diagram
├── tlio_model_data.h          # Embedded C header for STM32H7 deployment (~668 KB)
├── ardupilot_GHOST/           # ArduPilot integration (AP_GHOST C++ library)
└── paper/                     # arXiv paper sources
```

---

## Quickstart

### 1. Set up the environment
```bash
pip install tensorflow==2.19 numpy matplotlib pandas
```

### 2. Run inference with the pre-trained model
```python
import tensorflow as tf
from train_tf import nll_loss, displacement_mae
from load_data_tf import load_tlio_sequence, create_windows, WINDOW_SIZE, STRIDE

model = tf.keras.models.load_model(
    'GHOST_v25_TFLM.keras',
    custom_objects={'nll_loss': nll_loss, 'displacement_mae': displacement_mae},
    compile=False,
)

# Load any tlio_golden sequence
imu, pos, quat = load_tlio_sequence('/path/to/tlio_golden/seq_id')
X, y = create_windows(imu, pos, quat, WINDOW_SIZE, STRIDE)

pred = model.predict(X[:1])      # → (1, 6)
displacement = pred[0, :3]       # [dx, dy, dz] body frame, metres
log_variance = pred[0, 3:]       # per-axis log-variance for EKF covariance
```

### 3. Reproduce the paper's figures
```bash
jupyter notebook lab_car_sim.ipynb
```

### 4. Train from scratch
Open `lab_tensor.ipynb`. Set the `tlio_base` path to your local copy of the dataset, then run all cells.

---

## ArduPilot integration

The `ardupilot_GHOST/` directory contains the C++ library that runs the model inside an ArduPilot firmware build. To enable on a CubeOrangePlus:

```
AHRS_EKF_TYPE  = 3    # Use EKF3
EK3_SRC1_POSXY = 6    # ExternalNav for horizontal position
EK3_SRC1_POSZ  = 1    # Barometer for altitude
GPS_TYPE       = 0    # Disable GPS receiver
VISO_TYPE      = 1    # Enable MAVLink vision-position stream
```

After flashing the GHOST-enabled firmware, the autopilot reports:
```
GHOST: WAVE_Navigation_system initialized
WAVE: Memory used 227024 / 1048576 bytes
EKF3 IMU0 is using external nav data
```

See `paper/main.tex` §3.5 for a full description of the integration.

---

## License

### Code
The source code in this repository is released under the **Apache License 2.0** — see [LICENSE](LICENSE) for the full text. You may use, modify, and distribute the code in commercial and non-commercial projects subject to the terms in that file.

### Pre-trained model weights — important caveat
The model weights distributed in this repository (`GHOST_v25_TFLM.keras`, `tlio_model_data.h`) were trained on the [`tlio_golden`](https://github.com/CathIAS/TLIO) dataset, which is licensed by its original authors under **CC BY-NC 4.0** (non-commercial). As a consequence:

- ✅ The **code** can be used commercially (Apache 2.0).
- ⚠️ The **released weights** are derivative of CC BY-NC training data; commercial deployment of the released weights is **not granted by this repository**.
- ✅ For commercial use, **retrain the network on a dataset that is compatible with your intended use** (e.g., your own collected data, or contact the `tlio_golden` authors for a commercial licence). The training pipeline (`lab_tensor.ipynb`, `model_arch_v25_tflm.py`, `train_tf.py`) is fully provided so you can do this with minimal effort.

If you use this work, please cite the paper (BibTeX below) and respect the upstream `tlio_golden` license for any model weights you produce.

---

## Citation

```bibtex
@misc{abass2026ghost,
  title         = {{GHOST}: A {GPS}-Free Inertial Navigation System for Autonomous Vehicles},
  author        = {Abass, Abdalhay},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

(The arXiv ID will be filled in once the preprint is accepted — typically ~24 hours after submission.)

---

## Building on this work

If you want to:
- **Train on your own IMU data** → adapt `load_data_tf.py::load_tlio_sequence()` to your data format; the rest of the pipeline is unchanged.
- **Deploy on different microcontrollers** → the model exports to a 668 KB `tlio_model_data.h` that fits any STM32H7-class platform with at least 512 KB free flash and 256 KB activation arena.
- **Integrate with PX4 or another autopilot** → adapt `ardupilot_GHOST/AP_GHOST.cpp` to publish position estimates in your stack's idiomatic format. The MAVLink `VISION_POSITION_ESTIMATE` messages produced by the library are autopilot-agnostic.

Issues and pull requests are welcome.

---

## Contact

**Abdalhay Abass** — Independent Researcher
📧 abdalhayabass@gmail.com

For commercial-licensing inquiries (custom training data, integration support, deployment consulting), please reach out via email.

---

## Acknowledgements

This work builds on the foundational [TLIO](https://cathias.github.io/TLIO/) research by Liu et al. (RAL 2020). The benchmark dataset `tlio_golden` is released by the same authors under CC BY-NC. The deployment side of GHOST uses the open-source [ArduPilot](https://ardupilot.org) autopilot suite and the [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) runtime.
