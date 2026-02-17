"""
Experiment 2: Qari-OCR v0.3
Model: NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
Size: ~2B parameters
Base: Qwen2-VL-2B-Instruct (fine-tuned for Arabic OCR)
Paper: https://arxiv.org/abs/2506.02295
Source: https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
"""

import time
import os
import torch
import pypdfium2 as pdfium
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def pdf_to_image(pdf_path, page_num=0, dpi=300):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    scale = dpi / 72
    pil_image = page.render(scale=scale).to_pil()
    print("[INFO] Image size: {}".format(pil_image.size))
    return pil_image


def main():
    print("=" * 60)
    print("  Experiment 2: Qari-OCR v0.3")
    print("  Arabic-specialized OCR by NAMAA")
    print("=" * 60)

    # --- Load Model ---
    model_name = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
    print("[INFO] Loading model: {}".format(model_name))
    start_load = time.time()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    load_time = time.time() - start_load
    print("[INFO] Model loaded in {:.1f}s".format(load_time))

    # --- Convert PDF to Image ---
    pdf_path = r"C:\Users\MSI1\Desktop\testing_ocr\وكالة الكترونية\وكالة الكترونية 009.pdf"
    image = pdf_to_image(pdf_path, dpi=300)

    temp_path = "temp_qari.png"
    image.save(temp_path)

    # --- Build Prompt ---
    prompt = (
        "Below is the image of one page of a document, "
        "as well as some raw textual content that was previously extracted for it. "
        "Just return the plain text representation of this document as if you were reading it naturally. "
        "Do not hallucinate."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file://{}".format(temp_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # --- Process Input ---
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # --- Run OCR ---
    print("[INFO] Running OCR...")
    start_ocr = time.time()

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    ocr_time = time.time() - start_ocr

    # --- Display Results ---
    print("\n" + "=" * 60)
    print("  RESULTS: Qari-OCR v0.3")
    print("=" * 60)
    print("  Device:    cpu")
    print("  Load time: {:.1f}s".format(load_time))
    print("  OCR time:  {:.1f}s".format(ocr_time))
    print("  Length:    {} chars".format(len(result)))
    print("=" * 60)
    print("\n" + result)
    print("\n" + "=" * 60)

    # --- Save Result ---
    os.makedirs("output", exist_ok=True)
    with open("output/exp2_qari_result.txt", "w", encoding="utf-8") as f:
        f.write("Experiment 2: Qari-OCR v0.3\n")
        f.write("Model: {}\n".format(model_name))
        f.write("Device: cpu\n")
        f.write("Load time: {:.1f}s\n".format(load_time))
        f.write("OCR time: {:.1f}s\n".format(ocr_time))
        f.write("Length: {} chars\n".format(len(result)))
        f.write("=" * 60 + "\n\n")
        f.write(result)

    print("[INFO] Result saved to output/exp2_qari_result.txt")

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == "__main__":
    main()