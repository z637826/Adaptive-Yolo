"""
YOLOv14 Demo Image Generator — v2
Generates paper figures from real-world scene images.
Usage: python pic/run_demo.py
Output: pic/demo_*.jpg
"""
import cv2, numpy as np, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))
from ultralytics import YOLO, __version__

DATA_DIR = Path("data")
OUT_DIR  = Path("pic")
MODEL_PATH = "yolov12n.pt"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"YOLO {__version__} | model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("Model loaded.\n")

SCENES = [
    ("fisheye_street.jpg",     "demo_fisheye.jpg",     "Fisheye / Wide-Angle"),
    ("game_screenshot.jpg",    "demo_game.jpg",        "Game Character"),
    ("drone_aerial.jpg",       "demo_drone.jpg",       "Drone Top-Down"),
    ("panorama.jpg",           "demo_panorama.jpg",    "360° Panorama"),
    ("street_people.jpg",      "demo_standard.jpg",    "Standard Street"),
]

for src_name, dst_name, label in SCENES:
    src = DATA_DIR / src_name
    if not src.exists():
        print(f"  SKIP {src_name} (not found)")
        continue

    img = cv2.imread(str(src))
    if img is None:
        print(f"  FAIL {src_name} (cannot read)")
        continue

    h, w = img.shape[:2]
    print(f"  {label}: {w}x{h} → ", end="")

    # Predict
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    annotated = results[0].plot()

    out = OUT_DIR / dst_name
    cv2.imwrite(str(out), annotated)
    print(f"saved ({os.path.getsize(out)//1024} KB)")

# === Paper Figure 2: 2x2 with labels ===
print("\nAssembling Figure 2…")
keys = ["demo_fisheye.jpg", "demo_game.jpg", "demo_drone.jpg", "demo_panorama.jpg"]
panels = [cv2.imread(str(OUT_DIR / k)) for k in keys]
panels = [p for p in panels if p is not None]

# Uniform height
H = 280
resized = []
for p in panels:
    s = H / p.shape[0]
    r = cv2.resize(p, (int(p.shape[1]*s), H))
    resized.append(r)

# Pad widths to align columns
max_w_per_col = [max(r.shape[1] for r in resized[i::2]) for i in range(2)]
padded = []
for i, r in enumerate(resized):
    target_w = max_w_per_col[i % 2]
    if r.shape[1] < target_w:
        r = cv2.copyMakeBorder(r, 0, 0, 0, target_w - r.shape[1],
                                cv2.BORDER_CONSTANT, value=(220,220,220))
    padded.append(r)

row0 = np.hstack(padded[:2])
row1 = np.hstack(padded[2:])
# Make rows equal width
max_rw = max(row0.shape[1], row1.shape[1])
if row0.shape[1] < max_rw:
    row0 = cv2.copyMakeBorder(row0, 0, 0, 0, max_rw-row0.shape[1],
                               cv2.BORDER_CONSTANT, value=(220,220,220))
if row1.shape[1] < max_rw:
    row1 = cv2.copyMakeBorder(row1, 0, 0, 0, max_rw-row1.shape[1],
                               cv2.BORDER_CONSTANT, value=(220,220,220))
full = np.vstack([row0, row1])

# Add label band
LH = 48
canvas = np.full((full.shape[0]+LH, full.shape[1], 3), 255, dtype=np.uint8)
canvas[LH:] = full

# Helper: draw text with background for readability
def put_label(img, text, x, y, color=(0,0,0), bg=(255,255,255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+4), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

put_label(canvas, "(a) Fisheye / Wide-Angle",  12, 35, (30,60,180))
put_label(canvas, "(b) Game Character",         padded[0].shape[1]+12, 35, (30,60,180))
put_label(canvas, "(c) Drone Top-Down",         12, LH+25, (30,60,180))
put_label(canvas, "(d) 360° Panorama",          padded[2].shape[1]+12, LH+25, (30,60,180))

fig2 = OUT_DIR / "demo_figure2.jpg"
cv2.imwrite(str(fig2), canvas)
print(f"Figure 2 saved ({os.path.getsize(fig2)//1024} KB)")

# === Supplementary figure: standard + 4 scene grid ===
print("\nAssembling supplementary figure…")
std = cv2.imread(str(OUT_DIR / "demo_standard.jpg"))
if std is not None:
    s = H / std.shape[0]
    std_r = cv2.resize(std, (int(std.shape[1]*s), H))
    cols = []
    for r in [std_r, resized[2]]:  # standard + drone
        rw = r.shape[1]
        if rw < max_w_per_col[0]:
            r = cv2.copyMakeBorder(r, 0,0,0, max_w_per_col[0]-rw,
                                    cv2.BORDER_CONSTANT, value=(220,220,220))
        cols.append(r)
    supp = np.hstack(cols)
    supp_path = OUT_DIR / "demo_supplementary.jpg"
    cv2.imwrite(str(supp_path), supp)
    print(f"Supplementary saved ({os.path.getsize(supp_path)//1024} KB)")

# === Copy to paper/ ===
print("\nCopying to paper/ …")
paper = Path(__file__).parent.parent / "paper"
for dst_name in ["demo_fisheye.jpg","demo_game.jpg","demo_drone.jpg",
                  "demo_panorama.jpg","demo_standard.jpg","demo_figure2.jpg"]:
    src_f = OUT_DIR / dst_name
    if src_f.exists():
        import shutil
        shutil.copy2(str(src_f), str(paper / dst_name))
        print(f"  {dst_name} → paper/")

print("\nDone!")
