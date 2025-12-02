import os
import io
import base64
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import interpolation as inter

# ============================================================
#  Tamil characters list (same order as your Colab training)
# ============================================================
tamil_characters = [
'அ', 'ஆ', 'ஓ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ',
'ஃ', 'ரூ', 'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ', 'னூ', 'ா', 'ெ', 'ே', 'க', 'ை', 'ஸ்ரீ',
'ஸு', 'ஷு', 'ஜு', 'ஹு', 'க்ஷு', 'ஸூ', 'ஷூ', 'ஜூ', 'ங', 'ஹூ', 'க்ஷூ', 'க்', 'ங்',
'ச்', 'ஞ்', 'ட்', 'ண்', 'த்', 'ந்', 'ச', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்',
'ள்', 'ற்', 'ன்', 'ஞ', 'ஸ்', 'ஷ்', 'ஜ்', 'ஹ்', 'க்ஷ்', 'ஔ', 'ட', 'ண', 'த', 'ந',
'இ', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ஈ', 'ஸ', 'ஷ', 'ஜ',
'ஹ', 'க்ஷ', 'கி', 'ஙி', 'சி', 'ஞி', 'டி', 'உ', 'ணி', 'தி', 'நி', 'பி', 'மி',
'யி', 'ரி', 'லி', 'வி', 'ழி', 'ஊ', 'ளி', 'றி', 'नि', 'ஸி', 'ஷி', 'ஜி', 'ஹி',
'க்ஷி', 'கீ', 'ஙீ', 'எ', 'சீ', 'ஞீ', 'டீ', 'ணீ', 'தீ', 'நீ', 'பீ', 'மீ', 'யீ',
'ரீ', 'ஏ', 'லீ', 'வீ', 'ழீ', 'ளீ', 'றீ', 'னீ', 'ஸீ', 'ஷீ', 'ஜீ', 'ஹீ', 'ஐ',
'க்ஷீ', 'கு', 'ஙு', 'சு', 'ஞு', 'டு', 'ணு', 'து', 'நு', 'பு', 'ஒ', 'மு', 'யு',
'ரு', 'லு', 'வு', 'ழு', 'ளு', 'று', 'னு', 'கூ'
]


# ============================================================
#  MODEL ARCHITECTURE — MUST MATCH YOUR COLAB TamilNet
# ============================================================
class TamilNet(nn.Module):
    def __init__(self):
        super(TamilNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, len(tamil_characters))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool1(F.relu(self.bn6(self.conv6(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================
#  ROBUST IMAGE PREPROCESSING (replaces previous preprocess)
# ============================================================
def determine_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
    return score


def preprocess_image_bytes_robust(image_bytes, target_size=(64,64), auto_invert=True):
    # decode bytes
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # 1) convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) denoise + median blur
    blur = cv2.medianBlur(gray, 5)
    denoise = cv2.fastNlMeansDenoising(blur, None, 17, 9, 17)

    # 3) adaptive threshold (binary)
    th = cv2.adaptiveThreshold(denoise, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    # 4) determine skew by scoring (same as your code)
    def score_for_angle(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        return np.sum((hist[1:] - hist[:-1]) ** 2)

    scores = [score_for_angle(th, a) for a in range(-5, 6)]
    best_angle = range(-5, 6)[int(np.argmax(scores))]

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), best_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.medianBlur(gray_rot, 5)
    denoise2 = cv2.fastNlMeansDenoising(blur2, None, 17, 9, 17)
    final = cv2.adaptiveThreshold(denoise2, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    # 5) auto-invert if needed:
    # If most pixels are white (mean>127) and model expects white foreground on black
    if auto_invert:
        mean_val = np.mean(final)
        # If image is mostly white (background white), invert to make foreground white on black
        # Adjust logic depending on what your model expects.
        if mean_val > 127:
            final = cv2.bitwise_not(final)

    # 6) find largest contour and crop to bounding box (tight)
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # pick largest area contour
        c = max(contours, key=cv2.contourArea)
        x,y,wc,hc = cv2.boundingRect(c)
        # add padding
        pad = int(max(wc, hc)*0.2)
        x0 = max(0, x-pad)
        y0 = max(0, y-pad)
        x1 = min(final.shape[1], x+wc+pad)
        y1 = min(final.shape[0], y+hc+pad)
        crop = final[y0:y1, x0:x1]
    else:
        crop = final

    # 7) make square by padding and center the symbol
    h2, w2 = crop.shape[:2]
    size = max(h2, w2)
    square = 255 * np.ones((size, size), dtype=np.uint8)  # white background
    # paste centered
    y_off = (size - h2) // 2
    x_off = (size - w2) // 2
    square[y_off:y_off+h2, x_off:x_off+w2] = crop

    # 8) convert to PIL and resize using BILINEAR to target_size
    pil = Image.fromarray(square).convert("L")
    pil = pil.resize(target_size, Image.BILINEAR)

    # 9) FINAL: optional invert to match training mean (you can remove if not needed)
    # Convert to tensor later using transforms in server code.

    return pil, square  # pil is what we'll feed torchvision transforms; 'square' for display if needed


# ============================================================
#  FLASK SERVER
# ============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_PATHS = ["tamilnet.pth", "models/tamilnet.pth"]

# Load model
model = TamilNet()
device = torch.device("cpu")

loaded = False
for path in MODEL_PATHS:
    if os.path.exists(path):
        print(f"Loading model from: {path}")
        state = torch.load(path, map_location=device)

        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

        loaded = True
        break

if not loaded:
    raise FileNotFoundError("❌ tamilnet.pth not found in root or models/ folder")

model.eval()


# ============================================================
#  ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    # USE THE ROBUST PREPROCESSING FUNCTION
    pil_img, processed = preprocess_image_bytes_robust(image_bytes)

    # Ensure single-channel L mode (grayscale) before transform
    pil_img = pil_img.convert("L")

    # Save the processed image for visual comparison (Colab vs Server)
    try:
        pil_img.save("last_server_processed.png")
    except Exception:
        pass

    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ---------- CREATE TENSOR ----------
    img_tensor = transform(pil_img).unsqueeze(0)

    # ---------- TENSOR SHA + STATS (for parity debugging) ----------
    arr = img_tensor.cpu().numpy().astype(np.float32)
    flat = arr.ravel()
    sha = hashlib.sha256(flat.tobytes()).hexdigest()
    print("SERVER TENSOR shape:", img_tensor.shape,
          "min/max:", img_tensor.min().item(), "/", img_tensor.max().item(),
          "mean/std:", img_tensor.mean().item(), "/", img_tensor.std().item())
    print("SERVER TENSOR SHA256:", sha)

    # ---------- DEBUG PRINTS (EXISTING) ----------
    print("\n============= DEBUG =============")
    print("IMG TENSOR SHAPE:", img_tensor.shape)
    print("DTYPE:", img_tensor.dtype)
    print("TENSOR MIN/MAX:", torch.min(img_tensor).item(), "/", torch.max(img_tensor).item())

    # Don't move these — they help compare Colab vs server output
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)

    print("TOP5 INDICES:", top5_catid.tolist())
    print("TOP5 PROBS:", (top5_prob*100).tolist())
    print("TOP5 LABELS:", [tamil_characters[i] for i in top5_catid.tolist()])
    print("=================================\n")
    # -----------------------------------

    # Build predictions for frontend
    predictions = []
    for i in range(5):
        predictions.append({
            "rank": i + 1,
            "label": tamil_characters[top5_catid[i]],
            "confidence": round(float(top5_prob[i]) * 100, 2)
        })

    _, buffer = cv2.imencode('.png', processed)
    processed_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "predictions": predictions,
        "processed_image": f"data:image/png;base64,{processed_b64}"
    })


# ============================================================
#  RUN SERVER
# ============================================================
if __name__ == "__main__":
    print("==========================================")
    print(" EpigraphAI Backend Running...")
    print(" Open: http://127.0.0.1:5000")
    print("==========================================")
    app.run(debug=True)
