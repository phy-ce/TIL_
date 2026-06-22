"""
Dominant Color Analyzer (K-means)
----------------------------------
Extracts N dominant colors from an image using K-means, prints the ratio (%)
each color occupies, and saves a palette image.

Folder layout (paths are resolved relative to this script, so cwd does not matter):
    color_analysis/
    ├── script/
    │   └── color_analyze.py   <- this file
    ├── target/                <- put input images here
    └── output/                <- palette images are saved here (auto-created)

Usage:
    python color_analyze.py logo.png        # reads target/logo.png
    python color_analyze.py logo.png 8      # extract 8 dominant colors
    python color_analyze.py                 # process EVERY image in target/

Requirements:
    pip install opencv-python numpy
"""

import os
import sys
import cv2
import numpy as np

# --- Resolve project folders relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)          # the color_analysis/ folder
TARGET_DIR = os.path.join(BASE_DIR, "target")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def analyze_colors(image_path, k=5):
    # 1) Read the image (OpenCV loads in BGR order)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Could not read image: {image_path}")
        return

    # 2) Convert BGR -> RGB (the order we are used to)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3) Downscale if too large, for speed (longest side -> 200px)
    h, w = img.shape[:2]
    scale = 200 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 4) Flatten (height, width, 3) -> (num_pixels, 3). Each row is one color (a point)
    pixels = img.reshape(-1, 3).astype(np.float32)

    # 5) Run K-means: group similar colors into k clusters
    #    criteria = (stop condition, max iterations, accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )

    # centers = the k dominant colors (cluster centers), labels = cluster id per pixel
    centers = centers.astype(int)

    # 6) Count how many pixels fell into each cluster, then compute ratios
    counts = np.bincount(labels.flatten(), minlength=k)
    ratios = counts / counts.sum()

    # 7) Sort by ratio, descending
    order = np.argsort(-ratios)

    name = os.path.basename(image_path)
    print(f"\n[{name}] {k} dominant colors\n" + "-" * 40)
    results = []
    for rank, i in enumerate(order, start=1):
        r, g, b = centers[i]
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        pct = ratios[i] * 100
        print(f"{rank}. {hex_code}  RGB({r:>3},{g:>3},{b:>3})  {pct:5.1f}%")
        results.append((centers[i], ratios[i]))

    # 8) Save a palette bar image into output/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_palette.png")
    save_palette(results, out_path)
    print("-" * 40)
    print(f"Palette saved -> {os.path.relpath(out_path, BASE_DIR)}\n")


def save_palette(results, out_path, width=600, height=120):
    """Draw color bars in ratio order and save as an image"""
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for color, ratio in results:
        bar_w = int(round(ratio * width))
        # RGB -> BGR (cv2.imwrite expects BGR, so swap channels)
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(palette, (x, 0), (x + bar_w, height), bgr, -1)
        x += bar_w
    cv2.imwrite(out_path, palette)


def resolve_input(arg):
    """If arg is just a filename, look inside target/. Otherwise use as given."""
    if os.path.isabs(arg) or os.path.dirname(arg):
        return arg
    return os.path.join(TARGET_DIR, arg)


if __name__ == "__main__":
    # Parse args: optional <image>, optional <num_colors>
    image_arg = None
    k = 5
    for a in sys.argv[1:]:
        if a.isdigit():
            k = int(a)
        else:
            image_arg = a

    if image_arg:
        # Analyze a single image
        analyze_colors(resolve_input(image_arg), k)
    else:
        # No image given -> process every image in target/
        if not os.path.isdir(TARGET_DIR):
            print(f"[Error] target folder not found: {TARGET_DIR}")
            sys.exit(1)
        files = [f for f in sorted(os.listdir(TARGET_DIR))
                 if f.lower().endswith(IMAGE_EXTS)]
        if not files:
            print(f"No images found in {TARGET_DIR}. Put some images there first.")
            sys.exit(0)
        print(f"Found {len(files)} image(s) in target/. Analyzing...")
        for f in files:
            analyze_colors(os.path.join(TARGET_DIR, f), k)
