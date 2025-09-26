# 🚗 Lane Detection with Tiny U-Net (TuSimple)

Semantic lane segmentation for autonomous driving using a lightweight **Tiny U-Net** in PyTorch.  
Pipeline includes **data preprocessing** (TuSimple JSON → masks), **training**, **evaluation**, and **deployment** (CLI + Gradio UI).

---

## ✨ Highlights
- **Model:** Tiny U-Net (base=32) for binary lane segmentation
- **Dataset:** [TuSimple Lane Detection](https://www.kaggle.com/datasets/manideep1108/tusimple)
- **Input size:** 512×256 (configurable)
- **Validation metrics:** mIoU ≈ **0.46**, Dice/F1 ≈ **0.62** (≈ **0.67** with threshold tuning)
- **Speed:** ~**15 FPS** @ 512×256 on GPU
- **Artifacts:** `unet_best.pt`, CLI inferencer, Gradio UI

---

## 📁 Project Structure
```
.
├── README.md              # this file
├── train_lane_unet.py     # training script
├── infer_lanes.py         # CLI inference: image/video/webcam
├── app.py                 # Gradio UI
├── requirements.txt       # minimal dependencies
└── notebooks/             # (optional) Colab notebook steps
```

---

## 🗂 Dataset
- Dataset: **TuSimple Lane Detection** (via Kaggle)
- JSON label files provide lane points per image
- Preprocessing: convert JSON → binary lane masks, resize to 512×256
- ~3,600 train/val pairs created
- Split: **90% train / 10% validation**

> **Credits:** TuSimple dataset authors. Use under their license terms.

---

## 🧠 Model & Training
- **Architecture:** Tiny U-Net (encoder–decoder CNN)
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=1e-3)
- **Batch size:** 6
- **Epochs:** 6
- **Augmentation:** random horizontal flip
- **Precision:** mixed precision (`torch.amp`)
- **Best checkpoint:** `unet_best.pt`

---

## 📊 Results
- **mIoU:** ~0.458  
- **Dice/F1:** ~0.618 (≈0.671 at threshold 70)  
- **Inference speed:** ~15 FPS on Colab GPU (512×256)  

**Qualitative:**  
- Works well on clear, straight lanes  
- Struggles with shadows, curves, and faint markings (typical for a tiny model + short training)

---

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Training (optional)
```bash
python train_lane_unet.py   --train_dir /path/to/split/train   --val_dir   /path/to/split/val   --out       ./unet_best.pt   --width 512 --height 256   --batch_size 6 --workers 2 --epochs 6
```

### 3. Inference (CLI)

**Image → overlay**
```bash
python infer_lanes.py --weights unet_best.pt   --mode image --input road.jpg --output road_overlay.jpg
```

**Video → overlay**
```bash
python infer_lanes.py --weights unet_best.pt   --mode video --input input.mp4 --output output.mp4
```

**Webcam preview**
```bash
python infer_lanes.py --weights unet_best.pt --mode webcam
```

---

## 🖥️ Simple UI (Gradio)

```bash
pip install gradio
python app.py
```

- Open http://127.0.0.1:7860
- **Image tab:** drop an image → see overlay
- **Video tab:** drop an MP4 → processed output

---

## 🔮 Future Work
- Train longer (15–20 epochs)  
- Use Dice loss or BCE+Dice combo  
- Stronger augmentations (brightness/contrast, perspective)  
- Larger models (DeepLabV3, UNet++, LaneNet, SCNN)  
- Post-processing (polyline fitting, lane tracking)

---

## 📄 Requirements
Minimal `req.txt`:
```
numpy
opencv-python
torch
tqdm
gradio   # for UI
```

---

## 👤 Author
Darshan U P — Lane Detection Tiny U-Net Project (training, evaluation, deployment).

---

## 📜 License & Acknowledgements
- Code: MIT   
- Dataset: TuSimple Lane Detection (under their license/terms)  
- Thanks: TuSimple authors, PyTorch team, Gradio team  

---

### 📚 Citation (for U-Net)
```
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```
