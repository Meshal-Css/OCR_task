# Arabic Document OCR Project

Local and private OCR system for extracting text from Arabic documents using local models.
All processing runs **100% locally** — no data is ever sent to any external server.

---

## Environment

- **Python:** 3.11
- **Package Manager:** Conda
- **Environment Name:** test_ocr
- **OS:** Windows
- **GPU:** GeForce MX350 (2GB VRAM, CUDA 10.2)
- **Colab:** Google Colab T4 (15GB VRAM) for large models

---

## Project Structure

```
testing_ocr/
├── main.py                      # CLI runner for OCR experiments
├── handwriting_pipeline.py      # Handwriting detection pipeline
├── test_easyocr.py             # EasyOCR basic test
├── test_easyocr_enhanced.py    # EasyOCR with preprocessing comparison
├── test_lightonocr.py          # LightOnOCR test
├── test_qari.py                # Qari-OCR test (Colab)
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── ocr_engine.py           # Unified OCR engine (4 models)
│   ├── preprocessor.py         # Image preprocessing (7 methods)
│   ├── handwriting_detector.py # YOLOv8 handwriting detection
│   └── wakala_parser.py
├── files/                      # Test documents (not on GitHub)
├── output/                     # Results (not on GitHub)
└── tests/
```

---

## Hardware Constraints

- GPU: MX350 with 2GB VRAM — too small for most AI models
- Models must run on CPU locally (slow) or on Google Colab T4 (free)
- 4-bit quantization required for 2B+ models on Colab

---

## Models Tested

### Experiment Results Summary

| # | Model | Size | Platform | Doc Type | Avg Conf / Score | Result |
|---|-------|------|----------|----------|-----------------|--------|
| 1A | LightOnOCR-2-1B | 1B | CPU local | Electronic | 6/10 | Good structure, weak Arabic words |
| 1B | LightOnOCR-2-1B (denoise) | 1B | CPU local | Electronic | 5.5/10 | Preprocessing didn't help |
| 2A | Qari-OCR v0.3 (4bit) | 2B | Colab T4 | Electronic | 5/10 | Better Arabic, missing data |
| 2B | Qari-OCR v0.3 (new prompt) | 2B | Colab T4 | Electronic | FAIL | Infinite looping |
| 2C | Qwen2-VL-2B (base) | 2B | Colab T4 | Electronic | FAIL | Refused to read |
| 2D | Qwen2-VL-2B (English prompt) | 2B | Colab T4 | Electronic | 3/10 | Read backwards, no HW separation |
| 2E | Qwen2-VL-2B (structured) | 2B | Colab T4 | Electronic | FAIL | Hallucinated fake content |
| 3 | Arabic-English-handwritten-OCR-v3 | 3B | Demo | Handwritten | FAIL | Infinite looping |
| 4 | YOLOv8n (detection only) | Small | CPU local | Handwritten | 4/10 | Found signatures only |
| 5 | Color separation (CV) | - | Colab | Handwritten | FAIL | Colors too similar |
| 6 | Stroke analysis (CV) | - | Colab | Handwritten | 5/10 | Found signatures/stamps only |
| 7A | EasyOCR (original) | Light | CPU local | Electronic | 75.0% conf | Good Arabic, broken numbers |
| 7B | EasyOCR (full preprocess) | Light | CPU local | Electronic | 76.1% conf | Best with full pipeline |
| 7C | EasyOCR (original) | Light | CPU local | Handwritten | 15.6% conf | FAIL on handwriting |

---

## Detailed Experiment Results

### Experiment 1A: LightOnOCR-2-1B (Original)
- **Date:** 2026-02-15
- **File:** وكالة الكترونية 009.pdf
- **Preprocessing:** None
- **Device:** CPU (local)
- **Load time:** 11.9s | **OCR time:** 845.6s (~14 min)
- **Output:** 2433 chars
- **Strengths:** Excellent structure detection (tables, sections, HTML output)
- **Weaknesses:** Arabic words weak (الموثقين→المونتين, مقفلة→مقللة)
- **Numbers:** Partially wrong (missing digits)
- **Score:** 6/10

### Experiment 1B: LightOnOCR-2-1B (Denoise)
- **Date:** 2026-02-15
- **Same file, added denoise + CLAHE preprocessing**
- **OCR time:** 966.7s (slower)
- **Output:** 2448 chars
- **Result:** Some words improved, others regressed
- **Conclusion:** Preprocessing does NOT help on clean electronic documents
- **Score:** 5.5/10

### Experiment 2A: Qari-OCR v0.3
- **Date:** 2026-02-16
- **Model:** NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
- **Device:** Colab T4 GPU, 4-bit quantization, 150 DPI
- **OCR time:** 58.1s (14x faster than LightOnOCR)
- **Output:** 1288 chars
- **Strengths:** Better Arabic (مقفلة✅, المتاحة✅, إلغاء✅)
- **Weaknesses:** Missing critical data (names, IDs, dates, tables)
- **Score:** 5/10

### Experiment 7: EasyOCR
- **Date:** 2026-02-16
- **Device:** CPU local (no GPU)
- **Load time:** 4.9s

#### 7A: Electronic Document (وكالة الكترونية)
- **OCR time:** 102.4s (DPI 300) / 86.1s (DPI 400)
- **Detections:** 68-72
- **Avg Confidence:** 75.0%
- **Strengths:** Arabic words good, full paragraphs readable
- **Weaknesses:** Dates/numbers broken, edge words truncated

#### 7B: EasyOCR Preprocessing Comparison (DPI 400)

| Method | Detections | Avg Conf% | High Conf (>50%) | Time(s) |
|--------|-----------|-----------|-----------------|---------|
| original | 72 | 75.0 | 63 | 122.1 |
| contrast | 71 | 74.4 | 62 | 119.9 |
| denoise | 70 | 75.3 | 61 | 109.7 |
| sharpen | 73 | 74.3 | 61 | 103.9 |
| binary | 67 | 76.3 | 55 | 126.4 |
| **full** | **73** | **76.1** | **64** | **106.2** |

**Best method: full** (denoise → CLAHE → sharpen → Otsu threshold)

#### 7C: Handwritten Document (عقد ايجار ورقي)
- **OCR time:** 114.7s
- **Detections:** 77
- **Avg Confidence:** 15.6%
- **Result:** FAILED — almost all text unreadable
- **Score:** 1.5/10

---

## Handwriting Detection Experiments

### YOLOv8n Handwriting Detection
- **Model:** armvectores/yolov8n_handwritten_text_detection
- **Device:** CPU local
- **Result on صك ورقي:** 97 regions at 25% confidence → 8 regions at 60%
- **Correctly found:** Signatures and stamps at bottom
- **Missed:** Handwritten text inside form fields
- **Conclusion:** Trained on English docs, not suitable for Arabic mixed documents

### Computer Vision Approaches (Colab)
- **Color separation:** Failed — printed and handwritten text same color
- **Stroke width analysis:** Partially worked — found signatures/stamps only
- **Conclusion:** CV approaches not sufficient for mixed Arabic documents

---

## Key Findings

1. **Electronic documents:** EasyOCR with full preprocessing gives best results (76.1% confidence)
2. **Handwritten documents:** No model tested so far works well (best: 15.6%)
3. **Preprocessing helps slightly** on electronic docs but NOT significantly
4. **DPI 400** better than 300 for EasyOCR
5. **Arabic handwriting is extremely challenging** for all tested models
6. **VLM models (2B+)** tend to hallucinate or loop on Arabic documents
7. **Detection-based approaches** (YOLO) miss handwriting inside form fields

---



---

## Next Steps

1. Test **Arabic-English-handwritten-OCR-v3** on Colab with proper settings
2. Test **HATFormer** (small, handwriting specialist)
3. Test **Surya OCR** locally
4. Consider **fine-tuning** Qari-OCR on Saudi legal documents
5. Build combined pipeline: detection → separation → OCR

---

## Setup

```bash
# Create environment
conda create -n test_ocr python=3.11
conda activate test_ocr

# Install dependencies
pip install easyocr
pip install torch torchvision
pip install git+https://github.com/huggingface/transformers
pip install pypdfium2 Pillow opencv-python numpy
pip install ultralytics huggingface_hub

# Run EasyOCR test
python test_easyocr_enhanced.py

# Run handwriting detection
python handwriting_pipeline.py --file "files/doc.pdf" --confidence 0.6 --save
```

---

## Notes

- All files in `files/` are confidential — never upload to Git
- All processing must be 100% local (except Colab for large models)
- No cloud APIs (OpenAI, Google Vision, etc.)
- Output files shared via OneDrive (not on GitHub)
- GitHub repo: https://github.com/Meshal-Css/OCR_task.git