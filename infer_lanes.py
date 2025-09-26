import argparse, os, cv2, numpy as np, torch, torch.nn as nn

# ----------------- Tiny U-Net (same as training) -----------------
class DoubleConv(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(i,o,3,padding=1), nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1), nn.ReLU(True)
        )
    def forward(self,x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1=DoubleConv(3,base); self.p=nn.MaxPool2d(2)
        self.e2=DoubleConv(base,base*2); self.e3=DoubleConv(base*2,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DoubleConv(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2);   self.d1=DoubleConv(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p(e1)); e3=self.e3(self.p(e2))
        d2=self.u2(e3); d2=self.d2(torch.cat([d2,e2],1))
        d1=self.u1(d2); d1=self.d1(torch.cat([d1,e1],1))
        return self.out(d1)

# ----------------- Inference helpers -----------------
def load_model(weights_path, device, base=32):
    model = TinyUNet(base=base).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_mask(model, bgr, device, in_size=(512,256), thr=70):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    inp = cv2.resize(rgb, in_size).astype("float32") / 255.0
    x = torch.from_numpy(inp.transpose(2,0,1))[None].to(device)
    with torch.no_grad():
        y = torch.sigmoid(model(x))[0,0].cpu().numpy()
    prob = cv2.resize((y*255).astype("uint8"), (w0, h0), interpolation=cv2.INTER_NEAREST)
    mask = (prob >= thr).astype(np.uint8) * 255
    return mask

def overlay(bgr, mask):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    over = rgb.copy()
    over[mask>0, 1] = 255  # boost green where mask=1
    out = cv2.addWeighted(rgb, 0.6, over, 0.4, 0)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ----------------- Runners -----------------
def run_image(model, device, src, dst, in_size, thr):
    img = cv2.imread(src); assert img is not None, f"Cannot read {src}"
    m = predict_mask(model, img, device, in_size, thr)
    out = overlay(img, m)
    cv2.imwrite(dst, out)
    print("Saved:", dst)

def run_video(model, device, src, dst, in_size, thr, fps=None, smooth=0.6):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"Cannot open {src}"
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(dst, fourcc, fps, (w,h))
    ema = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        m = predict_mask(model, frame, device, in_size, thr)
        if ema is None: ema = m.astype(np.float32)
        else: ema = smooth * ema + (1.0 - smooth) * m
        m_s = (ema > 127).astype(np.uint8) * 255
        out = overlay(frame, m_s)
        vw.write(out)
    cap.release(); vw.release()
    print("Saved:", dst)

def run_webcam(model, device, cam_index, in_size, thr):
    cap = cv2.VideoCapture(cam_index)
    assert cap.isOpened(), f"Cannot open webcam index {cam_index}"
    ema = None; smooth=0.6
    print("Press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        m = predict_mask(model, frame, device, in_size, thr)
        if ema is None: ema = m.astype(np.float32)
        else: ema = smooth * ema + (1.0 - smooth) * m
        m_s = (ema > 127).astype(np.uint8) * 255
        out = overlay(frame, m_s)
        cv2.imshow("Lane Detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Lane segmentation inference")
    ap.add_argument("--weights", default="unet_best.pt", help="Path to trained .pt weights")
    ap.add_argument("--input", help="Path to image/video. If omitted and --webcam is set, uses webcam.")
    ap.add_argument("--output", default="lane_out.mp4", help="Output file (image or video)")
    ap.add_argument("--mode", choices=["image","video","webcam"], default="image")
    ap.add_argument("--webcam", type=int, default=0, help="Webcam index (for mode=webcam)")
    ap.add_argument("--width", type=int, default=512, help="Network input width")
    ap.add_argument("--height", type=int, default=256, help="Network input height")
    ap.add_argument("--thr", type=int, default=70, help="Binarization threshold (0-255)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model  = load_model(args.weights, device, base=32)
    in_size = (args.width, args.height)

    if args.mode == "image":
        assert args.input, "Please provide --input image path"
        # If output not an image extension, force .jpg
        out = args.output
        if os.path.splitext(out)[1].lower() not in [".jpg",".jpeg",".png",".bmp"]:
            out = os.path.splitext(out)[0] + ".jpg"
        run_image(model, device, args.input, out, in_size, args.thr)

    elif args.mode == "video":
        assert args.input, "Please provide --input video path"
        out = args.output
        if os.path.splitext(out)[1].lower() != ".mp4":
            out = os.path.splitext(out)[0] + ".mp4"
        run_video(model, device, args.input, out, in_size, args.thr)

    elif args.mode == "webcam":
        run_webcam(model, device, args.webcam, in_size, args.thr)

if __name__ == "__main__":
    main()
