"""
Image Preprocessor
==================
Handles PDF to image conversion and image enhancement
for better OCR accuracy.

Usage:
    from src.preprocessor import Preprocessor

    prep = Preprocessor()
    image = prep.pdf_to_image("document.pdf", page_num=0, dpi=300)
    enhanced = prep.enhance(image, method="denoise")
"""

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image


class Preprocessor:
    """Image preprocessing pipeline for OCR."""

    METHODS = ["original", "contrast", "sharpen", "binary", "adaptive", "denoise", "full"]

    # ----------------------------------------------------------
    # PDF to Image
    # ----------------------------------------------------------

    def pdf_to_image(self, pdf_path, page_num=0, dpi=300):
        """
        Convert a PDF page to PIL Image.

        Args:
            pdf_path: path to PDF file
            page_num: page index (0-based)
            dpi: render resolution (higher = better quality, slower)

        Returns:
            PIL.Image
        """
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_num]
        scale = dpi / 72
        pil_image = page.render(scale=scale).to_pil()
        return pil_image

    def pdf_page_count(self, pdf_path):
        """Return number of pages in PDF."""
        pdf = pdfium.PdfDocument(pdf_path)
        return len(pdf)

    # ----------------------------------------------------------
    # Image Enhancement
    # ----------------------------------------------------------

    def enhance(self, image, method="original"):
        """
        Apply preprocessing to image.

        Args:
            image: PIL.Image
            method: one of METHODS

        Returns:
            PIL.Image (processed)
        """
        if method == "original":
            return image
        elif method == "contrast":
            return self._contrast(image)
        elif method == "sharpen":
            return self._sharpen(self._contrast(image))
        elif method == "binary":
            return self._binarize(image)
        elif method == "adaptive":
            return self._adaptive_binarize(image)
        elif method == "denoise":
            return self._contrast(self._denoise(image))
        elif method == "full":
            return self._sharpen(self._contrast(self._denoise(image)))
        else:
            print("[WARN] Unknown method: {}. Returning original.".format(method))
            return image

    # ----------------------------------------------------------
    # Internal Methods
    # ----------------------------------------------------------

    def _to_gray(self, img_array):
        if len(img_array.shape) == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return img_array

    def _contrast(self, image):
        """CLAHE contrast enhancement."""
        img_array = np.array(image)
        gray = self._to_gray(img_array)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)

    def _sharpen(self, image):
        """Sharpen text edges."""
        img_array = np.array(image)
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(sharpened)

    def _binarize(self, image, threshold=150):
        """Fixed threshold binarization."""
        img_array = np.array(image)
        gray = self._to_gray(img_array)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return Image.fromarray(binary)

    def _adaptive_binarize(self, image):
        """Adaptive threshold for uneven lighting."""
        img_array = np.array(image)
        gray = self._to_gray(img_array)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        return Image.fromarray(binary)

    def _denoise(self, image):
        """Remove noise from image."""
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(img_array, h=10)
        else:
            denoised = cv2.fastNlMeansDenoisingColored(img_array, h=10)
        return Image.fromarray(denoised)