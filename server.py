from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
from pdf2image import convert_from_bytes
import torch
import io
import json
import threading

app = FastAPI()

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    # Use flash attention if your GPU supports it (A10/A100/H100)
    # attn_implementation="flash_attention_2",
)
model.eval()

# Compile model graph once at startup for faster repeated inference
# Uncomment if on PyTorch 2.0+ and want ~20% faster inference after warmup
# model = torch.compile(model, mode="reduce-overhead")

PROMPT = """You are an expert invoice OCR extraction system. Your ONLY job is to read the image carefully and return a JSON object.

CRITICAL FAILURE MODES TO AVOID:
- Returning null for fields that ARE visible in the image
- Missing dates due to alternate labels
- Missing totals located at the bottom of the document
- Skipping line items when table borders are not visible
- Using default values like 0 or placeholder text when data is missing

════════════════════════════════════════
STEP 1 — FULL IMAGE SCAN (MANDATORY)
════════════════════════════════════════
Scan all zones carefully:

ZONE 1 → TOP-LEFT     : Logo, vendor name, letterhead  
ZONE 2 → TOP-RIGHT    : Invoice number, dates, reference numbers  
ZONE 3 → TOP-CENTER   : Document title (Invoice, Bill, Receipt)  
ZONE 4 → LEFT BLOCK   : Customer / billing details  
ZONE 5 → MIDDLE TABLE : Line items (ALL rows)  
ZONE 6 → BOTTOM RIGHT : Totals (subtotal, tax, grand total, balance due)  
ZONE 7 → FOOTER       : Notes, payment terms, bank details, tax info  

Do NOT skip any zone.

════════════════════════════════════════
STEP 2 — DATE EXTRACTION
════════════════════════════════════════
Look for:
Date, Invoice Date, Bill Date, Tax Date, Issue Date, Dated, Billing Date

Rules:
- Return exactly as printed (no reformatting)
- Prefer Invoice/Bill Date if multiple exist
- Never return null if any date is visible in ZONE 1–3

════════════════════════════════════════
STEP 3 — LINE ITEMS
════════════════════════════════════════
Extract ALL rows from ZONE 5.

Identify rows by alignment and spacing, even without borders.

Map columns:
- description → Item / Product / Service / Particulars
- qty → Quantity / Qty / Units
- unit_price → Rate / Price / Unit Price
- total → Amount / Line Total / Value

Rules:
- Extract ALL items
- Do NOT merge rows
- Ignore subtotal/tax/discount rows
- Use null only if a column is missing

════════════════════════════════════════
STEP 4 — TOTAL AMOUNT (STRICT)
════════════════════════════════════════
Look ONLY in ZONE 6.

Prioritize:
1. Grand Total
2. Total Due / Amount Due / Net Payable / Balance Due
3. Largest bold/right-aligned number at bottom

Rules:
- Extract exact number only (no currency symbols)
- NEVER return 0 unless explicitly shown as 0
- NEVER guess values
- If multiple totals exist, choose final payable amount
- If no valid total is found → return null

════════════════════════════════════════
STEP 5 — NOTE EXTRACTION (STRICT)
════════════════════════════════════════
Extract ONLY real text from ZONE 7:
- Payment terms
- Banking details
- Remarks
- Tax/legal info

Rules:
- Return exact text if present
- If nothing exists → return null
- DO NOT use placeholder text

════════════════════════════════════════
OUTPUT FORMAT (UNCHANGED)
════════════════════════════════════════

Return ONLY valid JSON:

{
  "invoice_number": "Invoice No / Bill No",
  "date": "invoice date",
  "reference_number": "Ref No / PO Number or null",
  "vendor_name": "seller name",
  "vendor_address": "seller address",
  "customer_name": "buyer name",
  "customer_address": "buyer address",
  "line_items": [
    {
      "description": "...",
      "qty": null,
      "unit_price": null,
      "total": null
    }
  ],
  "total_amount": null,
  "note": null
}

Return ONLY JSON. No explanations, no extra text.
"""


def load_image(file_bytes: bytes, content_type: str) -> Image.Image:
    if content_type == "application/pdf":
        # Lower DPI (150 vs 200) — still readable, ~2x faster PDF conversion
        pages = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
        return pages[0].convert("RGB")
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    # Resize large images — VLM doesn't need >1280px on longest side
    max_side = 1280
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _run_inference(inputs) -> str:
    with torch.no_grad():
        # max_new_tokens=512 — enough for detailed invoices with many line items
        # do_sample=False — greedy, faster and deterministic
        # use_cache=True  — KV cache reuse across decode steps
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _parse_json(result: str):
    # Strip markdown fences if present
    if result.startswith("```"):
        parts = result.split("```")
        result = parts[1] if len(parts) > 1 else result
        if result.startswith("json"):
            result = result[4:]
        result = result.strip()

    start = result.find("{")
    end   = result.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "Model did not return valid JSON", "raw": result}

    try:
        return json.loads(result[start:end])
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}", "raw": result[start:end]}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if not file_bytes:
        return {"error": "Uploaded file is empty"}

    try:
        image = load_image(file_bytes, file.content_type)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT}
        ]
    }]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # pin_memory=False avoids extra copy for GPU tensors
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)

        result = _run_inference(inputs)
        return _parse_json(result)

    except Exception as e:
        return {"error": f"Inference failed: {e}"}


# Warm-up endpoint — call this once after deploy to pre-load CUDA kernels
# so the very first real request isn't slow
@app.on_event("startup")
async def warmup():
    dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
    msgs  = [{"role": "user", "content": [
        {"type": "image", "image": dummy},
        {"type": "text",  "text": "say hi"}
    ]}]
    text  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inp = processor(text=[text], images=img_in, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=5, do_sample=False)


@app.get("/health")
def health():
    return {"status": "ok"}
