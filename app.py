import os, cv2, numpy as np, torch, torch.nn as nn
import gradio as gr

# ---------- Tiny U-Net (same as training) ----------
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

# ---------- Load model ----------
WEIGHTS = os.getenv("LANE_WEIGHTS", "unet_best.pt")  # set env var or keep default
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

_model = TinyUNet(base=32).to(DEVICE)
_state = torch.load(WEIGHTS, map_location=DEVICE)
_model.load_state_dict(_state)
_model.eval()

def _predict_mask(rgb, width, height, thr):
    h0, w0 = rgb.shape[:2]
    inp = cv2.resize(rgb, (width, height)).astype("float32")/255.0
    x = torch.from_numpy(inp.transpose(2,0,1))[None].to(DEVICE)
    with torch.no_grad():
        y = torch.sigmoid(_model(x))[0,0].cpu().numpy()
    prob = cv2.resize((y*255).astype("uint8"), (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (prob >= thr).astype(np.uint8)*255

def _overlay(bgr, mask):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    over = rgb.copy()
    over[mask>0, 1] = 255  # green lanes
    out = cv2.addWeighted(rgb, 0.6, over, 0.4, 0)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ---------- Gradio functions ----------
def infer_image(img, width, height, thr):
    """
    img: PIL.Image/np.array (RGB) from Gradio
    returns: overlay image (RGB)
    """
    if img is None:
        return None
    rgb = np.array(img)  # RGB uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = _predict_mask(rgb, width, height, thr)
    out  = _overlay(bgr, mask)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def infer_video(video_path, width, height, thr, smooth_alpha):
    """
    Processes an uploaded video file and returns the path to a processed mp4.
    """
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Could not open the uploaded video."
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "lane_out.mp4"
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    ema = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        m = _predict_mask(rgb, width, height, thr)
        if ema is None:
            ema = m.astype(np.float32)
        else:
            ema = smooth_alpha * ema + (1.0 - smooth_alpha) * m
        ms = (ema > 127).astype(np.uint8)*255
        out = _overlay(frame, ms)
        vw.write(out)
    cap.release(); vw.release()
    return out_path

with gr.Blocks(title="Lane Detection Demo") as demo:
    gr.Markdown("## Lane Detection — Tiny U-Net\nUpload an image or video. Tweak threshold if needed.")
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                img_in  = gr.Image(type="pil", label="Input image")
                img_out = gr.Image(type="numpy", label="Overlay", interactive=False)
            with gr.Row():
                width    = gr.Slider(256, 768, value=512, step=32, label="Model width")
                height   = gr.Slider(128, 512, value=256, step=16, label="Model height")
                thr      = gr.Slider(0, 255, value=70, step=1, label="Threshold")
                run_img  = gr.Button("Run")
            run_img.click(fn=infer_image, inputs=[img_in, width, height, thr], outputs=img_out)

        with gr.Tab("Video"):
            vid_in  = gr.Video(label="Upload MP4")
            vid_out = gr.Video(label="Processed MP4")
            with gr.Row():
                v_width   = gr.Slider(256, 768, value=512, step=32, label="Model width")
                v_height  = gr.Slider(128, 512, value=256, step=16, label="Model height")
                v_thr     = gr.Slider(0, 255, value=70, step=1, label="Threshold")
                v_alpha   = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Temporal smoothing α")
                run_vid   = gr.Button("Process Video")
            run_vid.click(fn=infer_video, inputs=[vid_in, v_width, v_height, v_thr, v_alpha], outputs=vid_out)

demo.launch(server_name="127.0.0.1", server_port=7860)
