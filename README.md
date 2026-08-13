# Real-ESRGAN — Super Resolution with CUDA (Python 3.13 Compatible)

Real-ESRGAN aims at developing **Practical Algorithms for General Image/Video Restoration**.

This repo is a working, fixed setup of [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with:
- ✅ Python 3.13 compatibility fixes
- ✅ CUDA 12.x support (PyTorch 2.6.0+cu124)
- ✅ Gradio Web UI (`app.py`)
- ✅ Fixed directory/NoneType handling in inference script

---

## 🖥️ System Requirements

| Requirement | Version |
|---|---|
| Python | 3.13 |
| CUDA | 12.3 / 12.4 |
| GPU | NVIDIA (tested on RTX 3050) |

---

## ⚡ Quick Setup

### 1. Clone this repo
```bash
git clone https://github.com/1MeetPatel/SIH.git
cd SIH
```

### 2. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install other dependencies
```bash
pip install opencv-python pillow tqdm numpy
```

### 4. Install BasicSR (Python 3.13 compatible way)

```bash
git clone https://github.com/xinntao/BasicSR.git
```

Then edit `BasicSR/setup.py` — replace the `get_version()` function with:
```python
def get_version():
    with open('VERSION', 'r') as f:
        return f.read().strip()
```

Then install:
```bash
pip install --no-build-isolation -e ./BasicSR
```

### 5. Install Real-ESRGAN package
```bash
pip install --no-build-isolation --no-deps -e .
```

### 6. Download model weights
```bash
curl -L -o weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

---

## 🚀 Run Inference

### CLI — Batch process `inputs/` folder
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results
```

### CLI — Single image with face enhancement
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0014.jpg -o results --face_enhance
```

### Web UI (Gradio)
```bash
pip install gradio
python app.py
```
Then open **http://localhost:7860** in your browser.

---

## 🎛️ Available Models

| Model | Description |
|---|---|
| `RealESRGAN_x4plus` | General 4x upscaling (default) |
| `RealESRGAN_x4plus_anime_6B` | Optimized for anime images |
| `RealESRGAN_x2plus` | 2x upscaling |
| `RealESRNet_x4plus` | Faster, lighter 4x model |
| `realesr-animevideov3` | Anime video restoration |
| `realesr-general-x4v3` | General scenes, small model |

---

## 📁 Project Structure

```
.
├── inference_realesrgan.py   # CLI inference script (fixed for Python 3.13)
├── inference_realesrgan_video.py
├── app.py                    # Gradio Web UI
├── setup.py                  # Fixed for Python 3.13
├── requirements.txt
├── realesrgan/               # Core package
├── inputs/                   # Sample input images
├── weights/                  # Put downloaded .pth weights here
└── results/                  # Output images saved here
```

---

## 🔧 Python 3.13 Fix Applied

The `get_version()` function in `setup.py` used `exec()` + `locals()` which breaks in Python 3.13. Fixed to read directly from `VERSION` file.

---

## Credits

Original paper & code: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)  
Paper: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
