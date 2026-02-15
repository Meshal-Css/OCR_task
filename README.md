# Arabic Document OCR Project

Local and private OCR system for extracting text from Arabic documents using local models.
All processing runs **100% locally** — no data is ever sent to any external server.

---

## Environment

- **Python:** 3.11
- **Package Manager:** Conda
- **Environment Name:** test_ocr
- **OS:** Windows
- **GPU:** GeForce MX350 (2GB VRAM)
- **CUDA:** 10.2

---

## Project Structure

```
testing_ocr/
├── main.py
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── ocr_engine.py
│   ├── preprocessor.py
│   └── wakala_parser.py
├── files/
├── output/
└── tests/
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=5.0.0
pypdfium2>=4.0.0
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0
openpyxl>=3.1.0
```

Note: transformers must be installed from source for LightOnOCR-2 support:
```
pip install git+https://github.com/huggingface/transformers
```

---

## Hardware Constraints

- GPU: MX350 with 2GB VRAM
- Models must be under ~1.5GB to fit in VRAM
- Larger models run on CPU (slower)

---

## Models Under Evaluation

| # | Model | Size | Arabic | Status |
|---|-------|------|--------|--------|
| 1 | LightOnOCR-2-1B | 1B | Partial | Tested |
| 2 | GOT-OCR2.0 | 580M | Partial | Pending |
| 3 | Qari-OCR (NAMAA) | ~2B | Fine-tuned | Pending |
| 4 | DIMI-Arabic-OCR | ~3B | Fine-tuned | Pending |
| 5 | Arabic-English-handwritten-OCR-v3 | ~3B | Fine-tuned | Pending |
| 6 | PaddleOCR-VL | 0.9B | 109 langs | Pending |
| 7 | arabic-large-nougat | <1B | Specialized | Pending |

---

## Experiments

### Experiment 1A: LightOnOCR-2-1B (Original)
- **Model:** lightonai/LightOnOCR-2-1B
- **Date:** 2026-02-15
- **File Tested:** وكالة الكترونية 009.pdf
- **Preprocessing:** None (original image)
- **Device:** CPU
- **Model Load Time:** 11.9s
- **OCR Time:** 845.6s
- **Output Length:** 2433 chars
- **Accuracy:** ~6/10
- **Notes:**
  - Structure detection (tables, sections) is excellent
  - Names and ID numbers mostly correct
  - Arabic word errors: الموثقين→المونتين, مقفلة→مقللة, أصالة→أصمالة
  - Missing digits in document number: 7009102083→٧٠٩١٢٠٨٣
  - Dates: Hijri and Miladi swapped

### Experiment 1B: LightOnOCR-2-1B (Denoise)
- **Model:** lightonai/LightOnOCR-2-1B
- **Date:** 2026-02-15
- **File Tested:** وكالة الكترونية 009.pdf
- **Preprocessing:** Denoise + CLAHE contrast
- **Device:** CPU
- **Model Load Time:** (cached)
- **OCR Time:** 966.7s
- **Output Length:** 2448 chars
- **Accuracy:** ~5.5/10
- **Notes:**
  - Some improvements: ممثل شركة corrected, يتطلب corrected
  - Some regressions: معتمدة→معتددة, الموثقين→المولقين
  - Preprocessing did not help significantly — document is already clean
  - Conclusion: problem is model weakness in Arabic, not image quality

### Experiment 1 Summary

| Metric | Original | Denoise |
|--------|----------|---------|
| OCR Time | 845.6s | 966.7s |
| Output Length | 2433 chars | 2448 chars |
| Accuracy | ~6/10 | ~5.5/10 |
| Structure | Excellent | Excellent |
| Arabic Words | Weak | Weak |
| Numbers | Partially wrong | Partially wrong |

**Conclusion:** LightOnOCR-2-1B has excellent document structure understanding but weak Arabic text accuracy. Not suitable for legal documents requiring precise numbers and names. Image preprocessing does not improve results on clean electronic documents.

---

## Notes

- All files in `files/` are confidential — never upload to Git
- All processing must be 100% local
- No cloud APIs (OpenAI, Google, etc.)
- Tesseract was rejected for security concerns
- GPU is limited (2GB) — prioritize small models or CPU inference
- Output results are shared via OneDrive (not on GitHub)