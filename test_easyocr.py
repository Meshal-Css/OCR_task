"""
EasyOCR Enhanced Test Script
============================
Test EasyOCR with multiple preprocessing methods
to find the best result for Arabic documents.

Usage:
    python test_easyocr_enhanced.py
"""

import easyocr
import time
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

# === Settings ===
file_path = "وكالة الكترونية/وكالة الكترونية 990.pdf"
page_num = 0
dpi = 400  # Higher DPI for better quality

# === Preprocessing Functions ===

def preprocess_original(img):
    """No processing - baseline."""
    return img

def preprocess_contrast(img):
    """CLAHE contrast enhancement."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def preprocess_denoise(img):
    """Remove noise then enhance contrast."""
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def preprocess_sharpen(img):
    """Sharpen + contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

def preprocess_binary(img):
    """Adaptive binarization - best for scanned docs."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def preprocess_full(img):
    """Full pipeline: denoise → contrast → sharpen → threshold."""
    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Step 2: Grayscale + CLAHE
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 3: Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Step 4: Otsu threshold
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


METHODS = {
    "original": preprocess_original,
    "contrast": preprocess_contrast,
    "denoise": preprocess_denoise,
    "sharpen": preprocess_sharpen,
    "binary": preprocess_binary,
    "full": preprocess_full,
}


# === Convert PDF to image ===
print("=" * 60)
print("  EasyOCR Enhanced Test")
print("  File: {}".format(file_path))
print("  DPI: {}".format(dpi))
print("=" * 60)

if file_path.lower().endswith(".pdf"):
    print("[INFO] Converting PDF to image...")
    pdf = pdfium.PdfDocument(file_path)
    page = pdf[page_num]
    image = page.render(scale=dpi / 72).to_pil()
    img_array = np.array(image)
    print("[INFO] Image: {}x{}".format(image.width, image.height))
else:
    image = Image.open(file_path)
    img_array = np.array(image)
    print("[INFO] Image: {}x{}".format(image.width, image.height))

# === Load EasyOCR ===
print("[INFO] Loading EasyOCR...")
start = time.time()
reader = easyocr.Reader(['ar', 'en'], gpu=False)
print("[INFO] Loaded in {:.1f}s".format(time.time() - start))

# === Run OCR with each method ===
all_results = {}

for method_name, method_func in METHODS.items():
    print("\n" + "-" * 60)
    print("[TEST] Method: {}".format(method_name))
    
    # Preprocess
    processed = method_func(img_array.copy())
    
    # Save preprocessed image
    proc_path = "easyocr_{}.png".format(method_name)
    Image.fromarray(processed).save(proc_path)
    
    # Run OCR
    start = time.time()
    results = reader.readtext(proc_path)
    ocr_time = time.time() - start
    
    # Stats
    confidences = [conf for _, _, conf in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    high_conf = len([c for c in confidences if c > 0.5])
    total_text = " ".join([text for _, text, _ in results])
    
    all_results[method_name] = {
        "detections": len(results),
        "avg_confidence": avg_conf,
        "high_conf_count": high_conf,
        "time": ocr_time,
        "total_chars": len(total_text),
        "results": results,
    }
    
    print("  Detections: {}".format(len(results)))
    print("  Avg confidence: {:.1f}%".format(avg_conf * 100))
    print("  High conf (>50%): {}".format(high_conf))
    print("  Total chars: {}".format(len(total_text)))
    print("  Time: {:.1f}s".format(ocr_time))

# === Comparison Summary ===
print("\n" + "=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
print("{:<12} {:<12} {:<12} {:<12} {:<10}".format(
    "Method", "Detections", "Avg Conf%", "High Conf", "Time(s)"
))
print("-" * 60)

best_method = None
best_score = 0

for name, data in all_results.items():
    score = data["avg_confidence"] * data["high_conf_count"]
    if score > best_score:
        best_score = score
        best_method = name
    
    print("{:<12} {:<12} {:<12.1f} {:<12} {:<10.1f}".format(
        name,
        data["detections"],
        data["avg_confidence"] * 100,
        data["high_conf_count"],
        data["time"]
    ))

print("-" * 60)
print("BEST METHOD: {}".format(best_method))
print("=" * 60)

# === Save best results ===
best = all_results[best_method]
output_file = "easyocr_best_result.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("EasyOCR Enhanced Results\n")
    f.write("=" * 60 + "\n")
    f.write("File: {}\n".format(file_path))
    f.write("Best Method: {}\n".format(best_method))
    f.write("DPI: {}\n".format(dpi))
    f.write("OCR Time: {:.1f}s\n".format(best["time"]))
    f.write("Detections: {}\n".format(best["detections"]))
    f.write("Avg Confidence: {:.1f}%\n".format(best["avg_confidence"] * 100))
    f.write("=" * 60 + "\n\n")
    
    for i, (bbox, text, conf) in enumerate(best["results"]):
        f.write("{:3d}. [{:.0f}%] {}\n".format(i + 1, conf * 100, text))

# === Save comparison ===
comp_file = "easyocr_comparison.txt"
with open(comp_file, "w", encoding="utf-8") as f:
    f.write("EasyOCR Preprocessing Comparison\n")
    f.write("=" * 60 + "\n")
    f.write("File: {}\n".format(file_path))
    f.write("DPI: {}\n\n".format(dpi))
    f.write("{:<12} {:<12} {:<12} {:<12} {:<10}\n".format(
        "Method", "Detections", "Avg Conf%", "High Conf", "Time(s)"
    ))
    f.write("-" * 60 + "\n")
    for name, data in all_results.items():
        f.write("{:<12} {:<12} {:<12.1f} {:<12} {:<10.1f}\n".format(
            name,
            data["detections"],
            data["avg_confidence"] * 100,
            data["high_conf_count"],
            data["time"]
        ))
    f.write("-" * 60 + "\n")
    f.write("BEST: {}\n".format(best_method))

print("\n[INFO] Saved: {}".format(output_file))
print("[INFO] Saved: {}".format(comp_file))