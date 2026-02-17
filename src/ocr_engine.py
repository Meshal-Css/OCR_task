"""
OCR Engine
==========
Handles model loading and text extraction from images.
Supports multiple OCR models with a unified interface.

Usage:
    from src.ocr_engine import OCREngine

    engine = OCREngine(model_name="lightonai/LightOnOCR-2-1B")
    result, elapsed = engine.run("image.png")

Supported Models:
    - lightonai/LightOnOCR-2-1B (printed text, tables)
    - sherif1313/Arabic-English-handwritten-OCR-v3 (handwritten Arabic)
    - NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct (Arabic OCR)
    - Qwen/Qwen2-VL-2B-Instruct (general VLM)
"""

import time
import torch
from PIL import Image


class OCREngine:
    """Unified OCR engine supporting multiple models."""

    SUPPORTED_MODELS = {
        "lightonocr": "lightonai/LightOnOCR-2-1B",
        "handwritten-ar": "sherif1313/Arabic-English-handwritten-OCR-v3",
        "qari-v03": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        "qwen2-vl": "Qwen/Qwen2-VL-2B-Instruct",
    }

    def __init__(self, model_name="lightonai/LightOnOCR-2-1B"):
        # Allow shorthand names
        if model_name in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model_name]
        else:
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
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16

        if "LightOnOCR" in self.model_name:
            self._load_lightonocr()
        elif "handwritten-OCR" in self.model_name or "Qwen2" in self.model_name:
            self._load_qwen2_vl()
        elif "Qari-OCR" in self.model_name:
            self._load_qari()
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

    def _load_qwen2_vl(self):
        """Load Qwen2-VL based models (handwritten-OCR-v3, Qwen2-VL)."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
            from transformers import AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )

    def _load_qari(self):
        """Load Qari-OCR model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )

    # ----------------------------------------------------------
    # OCR Execution
    # ----------------------------------------------------------

    def run(self, image, max_tokens=2048, prompt=None):
        """
        Run OCR on an image.

        Args:
            image: PIL.Image or str (file path)
            max_tokens: maximum output tokens
            prompt: custom prompt (None = use default for model)

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
        elif "handwritten-OCR" in self.model_name or "Qwen2" in self.model_name:
            return self._run_qwen2_vl(image, max_tokens, prompt)
        elif "Qari-OCR" in self.model_name:
            return self._run_qari(image, max_tokens, prompt)

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

    def _run_qwen2_vl(self, image, max_tokens, prompt=None):
        """Run Qwen2-VL / handwritten-OCR-v3 inference."""
        from qwen_vl_utils import process_vision_info

        start = time.time()

        if prompt is None:
            prompt = "اقرأ كل المحتوى النصي الموجود في الصورة"

        # Save temp image for qwen_vl_utils
        image.save("/tmp/_ocr_temp.png")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "file:///tmp/_ocr_temp.png"},
                {"type": "text", "text": prompt},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        elapsed = time.time() - start

        return result, elapsed

    def _run_qari(self, image, max_tokens, prompt=None):
        """Run Qari-OCR inference."""
        from qwen_vl_utils import process_vision_info

        start = time.time()

        if prompt is None:
            prompt = (
                "Below is the image of one page of a document, "
                "as well as some raw textual content that was previously extracted for it. "
                "Just return the plain text representation of this document "
                "as if you were reading it naturally. "
                "Do not hallucinate."
            )

        image.save("/tmp/_ocr_temp.png")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "file:///tmp/_ocr_temp.png"},
                {"type": "text", "text": prompt},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
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