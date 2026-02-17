"""
Handwriting Extraction Pipeline
================================
Complete pipeline for extracting handwritten text from documents:
1. Convert PDF to image
2. Detect handwritten regions (YOLOv8)
3. Extract/crop handwritten regions
4. Run OCR on each region
5. Save results

Usage:
    python handwriting_pipeline.py --file "files/contract.pdf" --save
    python handwriting_pipeline.py --file "files/contract.png" --save
    python handwriting_pipeline.py --folder "files/عقود ورقية/" --save
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime

from src.preprocessor import Preprocessor
from src.handwriting_detector import HandwritingDetector


def process_file(file_path, detector, preprocessor, ocr_model=None, save=False):
    """
    Full pipeline: detect handwriting → extract regions → OCR.

    Args:
        file_path: path to PDF or image file
        detector: loaded HandwritingDetector
        preprocessor: Preprocessor instance
        ocr_model: OCR model name for reading (None = detection only)
        save: whether to save results to output/

    Returns:
        dict with results
    """
    filename = os.path.basename(file_path)
    print("\n" + "=" * 60)
    print("[FILE] {}".format(filename))
    print("=" * 60)

    # Step 1: Load/convert image
    if file_path.lower().endswith(".pdf"):
        image = preprocessor.pdf_to_image(file_path, dpi=300)
        print("[STEP 1] PDF → Image: {}x{}".format(image.width, image.height))
    else:
        from PIL import Image
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        print("[STEP 1] Image loaded: {}x{}".format(image.width, image.height))

    # Step 2: Detect handwritten regions
    print("[STEP 2] Detecting handwritten regions...")
    regions = detector.detect(image)

    if not regions:
        print("[INFO] No handwritten text detected in this document.")
        return {
            "file": filename,
            "regions_found": 0,
            "regions": [],
            "ocr_results": [],
        }

    print("[INFO] Found {} handwritten region(s)".format(len(regions)))
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        print("  Region {}: ({},{}) → ({},{}) conf={:.1f}%".format(
            i + 1, x1, y1, x2, y2, r["confidence"] * 100
        ))

    # Step 3: Extract regions
    print("[STEP 3] Extracting handwritten regions...")
    crops = detector.extract_regions(image, regions)

    # Step 4: OCR on each region (if model specified)
    ocr_results = []
    if ocr_model:
        print("[STEP 4] Running OCR on handwritten regions...")
        from src.ocr_engine import OCREngine

        engine = OCREngine(model_name=ocr_model)
        engine.load()

        for i, crop in enumerate(crops):
            # Preprocess the crop for better OCR
            processed = preprocessor.enhance(crop, method="denoise")
            result, ocr_time = engine.run(processed)
            ocr_results.append({
                "region": i + 1,
                "text": result,
                "time": ocr_time,
                "size": "{}x{}".format(crop.width, crop.height),
            })
            print("  Region {}: {} chars in {:.1f}s".format(
                i + 1, len(result) if result else 0, ocr_time
            ))
    else:
        print("[STEP 4] Skipped OCR (no model specified, detection only)")

    # Step 5: Save results
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(filename)[0]
        out_dir = "output/handwriting/{}".format(base)
        os.makedirs(out_dir, exist_ok=True)

        # Save annotated image
        annotated_path = os.path.join(out_dir, "annotated_{}.png".format(timestamp))
        detector.draw_detections(image, regions, output_path=annotated_path)

        # Save cropped regions
        for i, crop in enumerate(crops):
            crop_path = os.path.join(out_dir, "region_{:02d}.png".format(i + 1))
            crop.save(crop_path)

        # Save text results
        result_path = os.path.join(out_dir, "results_{}.txt".format(timestamp))
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("Handwriting Extraction Results\n")
            f.write("=" * 60 + "\n")
            f.write("File: {}\n".format(filename))
            f.write("Date: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("Detector: {}\n".format(detector.model_path))
            f.write("Regions found: {}\n".format(len(regions)))
            f.write("=" * 60 + "\n\n")

            for i, r in enumerate(regions):
                x1, y1, x2, y2 = r["bbox"]
                f.write("--- Region {} ---\n".format(i + 1))
                f.write("Location: ({},{}) → ({},{})\n".format(x1, y1, x2, y2))
                f.write("Confidence: {:.1f}%\n".format(r["confidence"] * 100))

                if i < len(ocr_results):
                    f.write("OCR Time: {:.1f}s\n".format(ocr_results[i]["time"]))
                    f.write("Text:\n{}\n".format(ocr_results[i]["text"]))
                f.write("\n")

        print("\n[SAVED] Results in: {}".format(out_dir))

    return {
        "file": filename,
        "regions_found": len(regions),
        "regions": regions,
        "ocr_results": ocr_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract handwritten text from Arabic documents"
    )
    parser.add_argument("--file", help="Path to a single file (PDF or image)")
    parser.add_argument("--folder", help="Path to folder with files")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--ocr", default=None,
                        help="OCR model for reading detected regions "
                             "(e.g., lightonai/LightOnOCR-2-1B). "
                             "If not set, only detection is performed.")
    parser.add_argument("--save", action="store_true",
                        help="Save results to output/handwriting/")
    args = parser.parse_args()

    if not args.file and not args.folder:
        print("[ERROR] Provide --file or --folder")
        sys.exit(1)

    # Header
    print("=" * 60)
    print("  Handwriting Extraction Pipeline")
    print("  Detector: YOLOv8n (handwritten text)")
    if args.ocr:
        print("  OCR Model: {}".format(args.ocr))
    else:
        print("  OCR: Detection only (no OCR model)")
    print("  Confidence: {:.0f}%".format(args.confidence * 100))
    print("=" * 60)

    # Init
    preprocessor = Preprocessor()
    detector = HandwritingDetector(confidence=args.confidence)
    detector.load()

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

    print("[INFO] Processing {} file(s)".format(len(files)))

    # Process
    all_results = []
    total_start = time.time()

    for file_path in files:
        result = process_file(
            file_path, detector, preprocessor,
            ocr_model=args.ocr, save=args.save
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("{:<35} {:<10} {:<15}".format("File", "Regions", "OCR Texts"))
    print("-" * 60)
    total_regions = 0
    for r in all_results:
        ocr_count = len(r["ocr_results"])
        print("{:<35} {:<10} {:<15}".format(
            r["file"][:33], r["regions_found"],
            ocr_count if ocr_count > 0 else "N/A"
        ))
        total_regions += r["regions_found"]

    print("-" * 60)
    print("Total files: {}".format(len(all_results)))
    print("Total handwritten regions: {}".format(total_regions))
    print("Total time: {:.1f}s".format(total_time))
    print("=" * 60)


if __name__ == "__main__":
    main()