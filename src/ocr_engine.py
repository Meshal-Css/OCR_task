"""
OCR Engine
==========
Handles model loading and text extraction from images.
Supports multiple OCR models with a unified interface.

Usage:
    from src.ocr_engine import OCREngine

    engine = OCREngine(model_name="lightonai/LightOnOCR-2-1B")
    result, elapsed = engine.run("image.png")
"""

import time
import torch
from PIL import Image


class OCREngine:
    """Unified OCR engine supporting multiple models."""

    SUPPORTED_MODELS = {
        "lightonocr": "lightonai/LightOnOCR-2-1B",
        # Add more models here as we test them
        # "got-ocr2": "stepfun-ai/GOT-OCR2_0",
        # "qari-ocr": "NAMAA-Space/Qari-OCR",
    }

    def __init__(self, model_name="lightonai/LightOnOCR-2-1B"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self.load_time = None

    # ----------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------

    def load(self):
        """Load model and processor."""
        print("[INFO] Loading model: {}".format(self.model_name))
        start = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        if "LightOnOCR" in self.model_name:
            self._load_lightonocr()
        else:
            print("[ERROR] Model not supported yet: {}".format(self.model_name))
            return False

        self.load_time = time.time() - start
        print("[INFO] Model loaded on {} in {:.1f}s".format(
            self.device, self.load_time
        ))
        return True

    def _load_lightonocr(self):
        """Load LightOnOCR model."""
        from transformers import (
            LightOnOcrForConditionalGeneration,
            LightOnOcrProcessor,
        )
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_name)
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)

    # ----------------------------------------------------------
    # OCR Execution
    # ----------------------------------------------------------

    def run(self, image, max_tokens=2048):
        """
        Run OCR on an image.

        Args:
            image: PIL.Image or str (file path)
            max_tokens: maximum output tokens

        Returns:
            tuple: (extracted_text, elapsed_seconds)
        """
        if self.model is None:
            print("[ERROR] Model not loaded. Call load() first.")
            return None, 0

        if isinstance(image, str):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if "LightOnOCR" in self.model_name:
            return self._run_lightonocr(image, max_tokens)

        return None, 0

    def _run_lightonocr(self, image, max_tokens):
        """Run LightOnOCR inference."""
        start = time.time()

        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point()
            else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        result = self.processor.decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - start

        return result, elapsed

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def info(self):
        """Return model info as dict."""
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "load_time": self.load_time,
            "loaded": self.model is not None,
        }