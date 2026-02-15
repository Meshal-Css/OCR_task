"""
Arabic Document OCR - Main
===========================
Entry point for running OCR experiments.

Usage:
    python main.py --file "files/doc.pdf" --method original
    python main.py --file "files/doc.pdf" --method denoise
    python main.py --file "files/doc.pdf" --method all
    python main.py --folder "files/وكالة الكترونية/" --method denoise --save
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime

from src.preprocessor import Preprocessor
from src.ocr_engine import OCREngine


def process_file(file_path, engine, preprocessor, method="original", save=False):
    """Process a single file through the OCR pipeline."""
    filename = os.path.basename(file_path)
    print("\n" + "-" * 60)
    print("[FILE] {}".format(filename))
    print("[METHOD] {}".format(method))
    print("-" * 60)

    # Step 1: Convert PDF to image
    if file_path.lower().endswith(".pdf"):
        image = preprocessor.pdf_to_image(file_path, dpi=300)
        print("[INFO] PDF converted to image: {}".format(image.size))
    else:
        from PIL import Image
        image = Image.open(file_path)
        print("[INFO] Image loaded: {}".format(image.size))

    # Step 2: Preprocess
    processed = preprocessor.enhance(image, method=method)
    print("[INFO] Preprocessing applied: {}".format(method))

    # Step 3: Run OCR
    result, ocr_time = engine.run(processed)
    print("[INFO] OCR completed in {:.1f}s".format(ocr_time))
    print("[INFO] Output: {} chars".format(len(result)))

    # Step 4: Save if requested
    if save:
        os.makedirs("output", exist_ok=True)
        base = os.path.splitext(filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = "output/{}_{}.txt".format(base, timestamp)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("File: {}\n".format(filename))
            f.write("Method: {}\n".format(method))
            f.write("Model: {}\n".format(engine.model_name))
            f.write("Device: {}\n".format(engine.device))
            f.write("OCR time: {:.1f}s\n".format(ocr_time))
            f.write("Output length: {} chars\n".format(len(result)))
            f.write("=" * 60 + "\n\n")
            f.write(result)
        print("[INFO] Saved: {}".format(out_path))

    return {
        "file": filename,
        "method": method,
        "time": ocr_time,
        "length": len(result),
        "text": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Arabic Document OCR")
    parser.add_argument("--file", help="Path to a single file (PDF or image)")
    parser.add_argument("--folder", help="Path to folder with files")
    parser.add_argument("--method", default="original",
                        help="Preprocessing method: original, contrast, sharpen, "
                             "binary, adaptive, denoise, full, all")
    parser.add_argument("--model", default="lightonai/LightOnOCR-2-1B",
                        help="Model name")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--save", action="store_true", help="Save results to output/")
    args = parser.parse_args()

    if not args.file and not args.folder:
        print("[ERROR] Provide --file or --folder")
        sys.exit(1)

    # Init
    print("=" * 60)
    print("  Arabic Document OCR")
    print("  Model: {}".format(args.model))
    print("  Method: {}".format(args.method))
    print("=" * 60)

    preprocessor = Preprocessor()
    engine = OCREngine(model_name=args.model)
    engine.load()

    # Collect files
    files = []
    if args.file:
        files = [args.file]
    elif args.folder:
        for ext in ["*.pdf", "*.png", "*.jpg", "*.jpeg"]:
            files.extend(glob.glob(os.path.join(args.folder, ext)))
        files = sorted(files)

    if not files:
        print("[ERROR] No files found")
        sys.exit(1)

    print("[INFO] Files: {}".format(len(files)))

    # Determine methods
    if args.method == "all":
        methods = Preprocessor.METHODS
    else:
        methods = [args.method]

    # Process
    all_results = []
    for file_path in files:
        for method in methods:
            result = process_file(
                file_path, engine, preprocessor,
                method=method, save=args.save
            )
            all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("{:<30} {:<12} {:<10} {:<8}".format("File", "Method", "Time(s)", "Chars"))
    print("-" * 60)
    for r in all_results:
        print("{:<30} {:<12} {:<10.1f} {:<8}".format(
            r["file"][:28], r["method"], r["time"], r["length"]
        ))
    print("=" * 60)


if __name__ == "__main__":
    main()