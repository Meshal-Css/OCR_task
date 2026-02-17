"""
Handwriting Detector
====================
Detects and extracts handwritten text regions from document images.
Uses YOLOv8 fine-tuned for handwriting detection to separate
handwritten text from printed text.

Usage:
    from src.handwriting_detector import HandwritingDetector

    detector = HandwritingDetector()
    regions = detector.detect("document.png")
    crops = detector.extract_regions("document.png", regions)

Models:
    - armvectores/yolov8n_handwritten_text_detection (default, lightweight)
"""

import os
import time
from PIL import Image


class HandwritingDetector:
    """Detect handwritten text regions in document images."""

    SUPPORTED_MODELS = {
        "yolov8n": "armvectores/yolov8n_handwritten_text_detection",
    }

    def __init__(self, model_name="yolov8n", confidence=0.25):
        """
        Args:
            model_name: key from SUPPORTED_MODELS or full HuggingFace path
            confidence: minimum detection confidence (0.0 - 1.0)
        """
        if model_name in self.SUPPORTED_MODELS:
            self.model_path = self.SUPPORTED_MODELS[model_name]
        else:
            self.model_path = model_name

        self.confidence = confidence
        self.model = None
        self.load_time = None

    # ----------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------

    def load(self):
        """Load YOLOv8 model from HuggingFace."""
        print("[INFO] Loading handwriting detector: {}".format(self.model_path))
        start = time.time()

        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            # Download model weights from HuggingFace
            model_file = hf_hub_download(
                repo_id=self.model_path,
                filename="best.pt"
            )
            self.model = YOLO(model_file)
            self.load_time = time.time() - start
            print("[INFO] Detector loaded in {:.1f}s".format(self.load_time))
            return True

        except Exception as e:
            print("[ERROR] Failed to load detector: {}".format(e))
            return False

    # ----------------------------------------------------------
    # Detection
    # ----------------------------------------------------------

    def detect(self, image, padding=10):
        """
        Detect handwritten text regions in an image.

        Args:
            image: PIL.Image or str (file path)
            padding: extra pixels around each detection box

        Returns:
            list of dicts with keys:
                - bbox: (x1, y1, x2, y2) coordinates
                - confidence: detection confidence
                - label: class label
        """
        if self.model is None:
            print("[ERROR] Model not loaded. Call load() first.")
            return []

        if isinstance(image, str):
            image = Image.open(image)

        start = time.time()
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )
        elapsed = time.time() - start

        regions = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            w, h = image.size

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = results[0].names.get(cls, "handwritten")

                # Apply padding
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(w, int(x2) + padding)
                y2 = min(h, int(y2) + padding)

                regions.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": round(conf, 3),
                    "label": label,
                })

        # Sort top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        print("[INFO] Detected {} handwritten regions in {:.2f}s".format(
            len(regions), elapsed
        ))
        return regions

    # ----------------------------------------------------------
    # Region Extraction
    # ----------------------------------------------------------

    def extract_regions(self, image, regions):
        """
        Crop detected handwritten regions from the image.

        Args:
            image: PIL.Image or str (file path)
            regions: list of dicts from detect()

        Returns:
            list of PIL.Image crops
        """
        if isinstance(image, str):
            image = Image.open(image)

        crops = []
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        return crops

    def save_regions(self, image, regions, output_dir="output/regions"):
        """
        Save detected regions as individual image files.

        Args:
            image: PIL.Image or str (file path)
            regions: list of dicts from detect()
            output_dir: directory to save cropped images

        Returns:
            list of saved file paths
        """
        if isinstance(image, str):
            image = Image.open(image)

        os.makedirs(output_dir, exist_ok=True)
        crops = self.extract_regions(image, regions)
        saved_paths = []

        for i, crop in enumerate(crops):
            conf = regions[i]["confidence"]
            path = os.path.join(output_dir, "hw_region_{:02d}_{:.0f}.png".format(
                i + 1, conf * 100
            ))
            crop.save(path)
            saved_paths.append(path)
            print("[INFO] Saved: {} ({}x{})".format(
                path, crop.width, crop.height
            ))

        return saved_paths

    def draw_detections(self, image, regions, output_path=None):
        """
        Draw bounding boxes on the image showing detected regions.

        Args:
            image: PIL.Image or str (file path)
            regions: list of dicts from detect()
            output_path: if set, save annotated image

        Returns:
            PIL.Image with drawn boxes
        """
        if isinstance(image, str):
            image = Image.open(image)

        from PIL import ImageDraw, ImageFont

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region["bbox"]
            conf = region["confidence"]

            # Draw red rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label
            label = "HW #{} ({:.0f}%)".format(i + 1, conf * 100)
            draw.text((x1, y1 - 15), label, fill="red")

        if output_path:
            annotated.save(output_path)
            print("[INFO] Annotated image saved: {}".format(output_path))

        return annotated

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def info(self):
        """Return detector info as dict."""
        return {
            "model": self.model_path,
            "confidence": self.confidence,
            "load_time": self.load_time,
            "loaded": self.model is not None,
        }