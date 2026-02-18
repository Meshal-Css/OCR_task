"""
TrOCR Arabic Handwriting Test
==============================
Test David-Magdy/TR_OCR_LARGE on Arabic handwritten documents.
Runs on CPU.

Usage:
    python test_trocr.py
"""

import os
import time
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pypdfium2 as pdfium

# === Settings ===
file_path = "عقد ايجار ورقي 23.pdf"
model_name = "David-Magdy/TR_OCR_LARGE"

# === Output folder ===
out_dir = "output/{}".format(model_name.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)

# === Load image ===
print("=" * 60)
print("  TrOCR Arabic Handwriting Test")
print("  Model: {}".format(model_name))
print("  File: {}".format(file_path))
print("  Output: {}".format(out_dir))
print("=" * 60)

if file_path.lower().endswith(".pdf"):
    pdf = pdfium.PdfDocument(file_path)
    page = pdf[0]
    image = page.render(scale=300 / 72).to_pil()
else:
    image = Image.open(file_path).convert("RGB")

print("[INFO] Image: {}x{}".format(image.width, image.height))

# === Load model ===
print("[INFO] Loading model (first time downloads ~1.5GB)...")
start = time.time()
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
load_time = time.time() - start
print("[INFO] Model loaded in {:.1f}s".format(load_time))

# === Method 1: Full image (single pass) ===
print("\n[TEST 1] Full image - single pass")
start = time.time()
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values, max_new_tokens=256)
result_full = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
t1 = time.time() - start
print("  Time: {:.1f}s".format(t1))
print("  Result: {}".format(result_full))

# === Method 2: Split into horizontal strips (line-by-line) ===
print("\n[TEST 2] Line-by-line (horizontal strips)")
w, h = image.size
strip_height = h // 15
results_lines = []

start = time.time()
for i in range(15):
    y1 = i * strip_height
    y2 = min((i + 1) * strip_height, h)
    strip = image.crop((0, y1, w, y2))

    pixel_values = processor(images=strip, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if text.strip():
        results_lines.append(text.strip())
        print("  Strip {:2d}: {}".format(i + 1, text.strip()))

t2 = time.time() - start

# === Method 3: Right half only (where handwriting usually is) ===
print("\n[TEST 3] Right half strips (handwritten fields area)")
right_half = image.crop((w // 3, 0, w, h))
rw, rh = right_half.size
strip_h = rh // 20
results_right = []

start = time.time()
for i in range(20):
    y1 = i * strip_h
    y2 = min((i + 1) * strip_h, rh)
    strip = right_half.crop((0, y1, rw, y2))

    pixel_values = processor(images=strip, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if text.strip():
        results_right.append(text.strip())
        print("  Strip {:2d}: {}".format(i + 1, text.strip()))

t3 = time.time() - start

# === Summary ===
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print("Model: {}".format(model_name))
print("Load time: {:.1f}s".format(load_time))
print("\nTest 1 (full image): {:.1f}s".format(t1))
print("  Result: {}".format(result_full[:100]))
print("\nTest 2 (line strips): {:.1f}s".format(t2))
print("  Lines found: {}".format(len(results_lines)))
for line in results_lines:
    print("  > {}".format(line))
print("\nTest 3 (right half strips): {:.1f}s".format(t3))
print("  Lines found: {}".format(len(results_right)))
for line in results_right:
    print("  > {}".format(line))

# === Save results ===
out_file = os.path.join(out_dir, "trocr_result.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("TrOCR Arabic Handwriting Test\n")
    f.write("=" * 60 + "\n")
    f.write("Model: {}\n".format(model_name))
    f.write("File: {}\n".format(file_path))
    f.write("Load time: {:.1f}s\n\n".format(load_time))
    f.write("--- Test 1: Full image ({:.1f}s) ---\n{}\n\n".format(t1, result_full))
    f.write("--- Test 2: Line strips ({:.1f}s) ---\n".format(t2))
    for line in results_lines:
        f.write("{}\n".format(line))
    f.write("\n--- Test 3: Right half strips ({:.1f}s) ---\n".format(t3))
    for line in results_right:
        f.write("{}\n".format(line))

print("\n[INFO] Saved: {}".format(out_file))
print("=" * 60)