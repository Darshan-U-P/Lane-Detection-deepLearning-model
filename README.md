Lane Detection with Tiny U-Net (TuSimple)

Semantic lane segmentation for autonomous driving using a lightweight Tiny U-Net in PyTorch.
Pipeline includes data preprocessing (TuSimple JSON → masks), training, evaluation, and deployment (CLI + Gradio UI).

✨ Highlights

Model: Tiny U-Net (base=32) for binary segmentation (lane vs background)

Dataset: TuSimple highway driving (Kaggle)

Input size: 512×256 (configurable)

Val metrics: mIoU ≈ 0.46, Dice/F1 ≈ 0.62 (≈ 0.67 with threshold tuning)

Speed: ~15 FPS @ 512×256 on GPU

Artifacts: unet_best.pt, CLI inferencer, Gradio UI

📁 Project Structure
.
├── README.md
├── train_lane_unet.py      # training script
├── infer_lanes.py          # CLI inference: image/video/webcam
├── app.py                  # Gradio UI
├── requirements.txt        # minimal deps for local run
└── (optional) notebooks/   # your Colab steps, if you add them

🧠 How it Works (Short)

Preprocess TuSimple: convert lane point annotations (JSON) → binary masks by drawing polylines; resize to 512×256; split 90/10 train/val.

Train Tiny U-Net (BCEWithLogitsLoss + Adam) for 6 epochs, mixed precision.

Evaluate with IoU/Dice; tune probability threshold (best ≈ 70).

Deploy with a Python CLI + Gradio app for images/videos.

🗂 Dataset

TuSimple Lane Detection (Kaggle).

You can download in Colab via:

import kagglehub
path = kagglehub.dataset_download("manideep1108/tusimple")
print(path)  # -> /root/.cache/kagglehub/datasets/.../versions/5


The preprocessing script walks the dataset, draws masks from labels, and saves pairs to /content/tmp_lane/{images,masks} during training.

Credits: TuSimple dataset authors. Use under their license terms.

🚀 Quickstart (Colab)

Preprocess → Train → Download model
Use the provided Colab cells (or adapt from train_lane_unet.py) to:

build ~3,600 image/mask pairs

train Tiny U-Net for 6 epochs

download unet_best.pt

Optional: Evaluate
Compute IoU/Dice and find the best binarization threshold (≈ 70).

🧪 Train Locally (optional)

Requires a machine with Python 3.9+ (CUDA optional).

python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt


Train (expects the split you created in Colab or locally):

python train_lane_unet.py \
  --train_dir /path/to/split/train \
  --val_dir   /path/to/split/val \
  --out       ./unet_best.pt \
  --width 512 --height 256 \
  --batch_size 6 --workers 2 --epochs 6


Key flags:

--base 16 for an even smaller/faster model

--no_amp to disable mixed precision

--cpu to force CPU training

🔍 Inference (CLI)

Image → overlay

python infer_lanes.py --weights unet_best.pt \
  --mode image --input road.jpg --output road_overlay.jpg \
  --width 512 --height 256 --thr 70


Video → overlay MP4

python infer_lanes.py --weights unet_best.pt \
  --mode video --input input.mp4 --output output.mp4 \
  --width 512 --height 256 --thr 70


Webcam preview (press q to quit)

python infer_lanes.py --weights unet_best.pt --mode webcam --thr 70


Tips:

Lower --width/--height (e.g., 384×192) for faster CPU inference

Adjust --thr (50–90) if overlays look too thin/too thick

🖥️ Simple UI (Gradio)
pip install gradio
python app.py


Open the printed URL (http://127.0.0.1:7860
)

Image tab: drop an image → Run

Video tab: drop an MP4 → Process (temporal smoothing included)

Set custom weights:

LANE_WEIGHTS=/path/to/unet_best.pt python app.py

📊 Results (Validation)

mIoU: ~0.458

Dice/F1: ~0.618 (≈ 0.671 at threshold 70)

FPS: ~15 on a Colab GPU at 512×256

Qualitative: Good on clear, straight lanes; struggles with shadows, curves, and faint markings (typical for a tiny model + short training).

⚙️ Training Details

Model: Tiny U-Net (base=32)

Loss: BCEWithLogitsLoss

Optimizer: Adam (lr=1e-3)

Batch size: 6

Epochs: 6

Augmentation: random horizontal flip

Precision: mixed (torch.amp)

Best checkpoint: unet_best.pt

🔮 Roadmap / Improvements

Longer training (15–20+ epochs)

Loss tweaks: Dice or BCE + Dice

Stronger augmentations (brightness/contrast, perspective)

Larger/backbone models (DeepLabV3, UNet++), or lane-specific methods (SCNN, LaneNet)

Post-processing (polyline fitting, lane tracking, perspective transform)

📄 Requirements

requirements.txt (minimal):

numpy
opencv-python
torch
tqdm
gradio  # only for the UI

📝 License & Acknowledgements

Code: choose a license (MIT recommended).

Dataset: TuSimple Lane Detection — follow dataset’s license/terms.

Acknowledgements: TuSimple authors; PyTorch & Gradio teams.

👤 Author

Darshan — lane detection Tiny U-Net project (training, evaluation, and deployment).

Citation (optional)

If you cite U-Net in your report:

@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
