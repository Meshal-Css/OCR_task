"""
Experiment 1: LightOnOCR-2-1B
Model: lightonai/LightOnOCR-2-1B
Size: 1B parameters
Type: End-to-end VLM for OCR
Source: https://huggingface.co/lightonai/LightOnOCR-2-1B
"""

import time
import os
import torch
import pypdfium2 as pdfium
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name="lightonai/LightOnOCR-2-1B"):
    print("[INFO] Loading model: {}".format(model_name))
    start = time.time()

    device = get_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    processor = LightOnOcrProcessor.from_pretrained(model_name)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)

    elapsed = time.time() - start
    print("[INFO] Model loaded on {} in {:.1f}s".format(device, elapsed))
    return model, processor, device, dtype, elapsed


def pdf_to_image(pdf_path, page_num=0, dpi=200):
    print("[INFO] Converting PDF page {} to image...".format(page_num + 1))
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    scale = dpi / 72
    pil_image = page.render(scale=scale).to_pil()
    print("[INFO] Image size: {}".format(pil_image.size))
    return pil_image


def run_ocr(model, processor, image, device, dtype):
    print("[INFO] Running OCR...")
    start = time.time()

    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    result = processor.decode(generated_ids, skip_special_tokens=True)

    elapsed = time.time() - start
    print("[INFO] OCR completed in {:.1f}s".format(elapsed))
    return result, elapsed


def save_result(result, load_time, ocr_time, device, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Experiment 1: LightOnOCR-2-1B\n")
        f.write("Device: {}\n".format(device))
        f.write("Model load time: {:.1f}s\n".format(load_time))
        f.write("OCR time: {:.1f}s\n".format(ocr_time))
        f.write("Output length: {} chars\n".format(len(result)))
        f.write("=" * 60 + "\n\n")
        f.write(result)
    print("[INFO] Result saved to {}".format(output_path))


def main():
    print("=" * 60)
    print("  Experiment 1: LightOnOCR-2-1B")
    print("  Task: Arabic Document OCR")
    print("=" * 60)

    # Load model
    model, processor, device, dtype, load_time = load_model()

    # Convert PDF to image
    pdf_path = r"C:\Users\MSI1\Desktop\testing_ocr\وكالة الكترونية\وكالة الكترونية 009.pdf"
    image = pdf_to_image(pdf_path, page_num=0, dpi=200)

    # Run OCR
    result, ocr_time = run_ocr(model, processor, image, device, dtype)

    # Display results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("  Device:          {}".format(device))
    print("  Model load time: {:.1f}s".format(load_time))
    print("  OCR time:        {:.1f}s".format(ocr_time))
    print("  Output length:   {} chars".format(len(result)))
    print("=" * 60)
    print("\n" + result)
    print("\n" + "=" * 60)

    # Save result
    save_result(result, load_time, ocr_time, device, "output/exp1_lightonocr_result.txt")


if __name__ == "__main__":
    main()