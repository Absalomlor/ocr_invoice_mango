import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import requests
import base64
import json
import re
import pandas as pd
import ast
import time

# --- CONFIG ---
GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
HEADERS = {'Content-Type': 'application/json'}

PROMPT = """
    You are an expert AI assistant specialized in comprehensive and highly accurate document data extraction. Your primary task is to process tax invoices with inconsistent and challenging layouts and extract ALL available information.

    **Challenge:** The documents have variable layouts. Information is often split across lines, and corresponding labels and values can be spatially distant. Simple OCR or fixed-template methods will fail. You must use contextual and spatial understanding to succeed.

    **Extraction Strategy (Hybrid Approach):**

    Your goal is to extract information into a structured JSON object for each invoice. We will use a hybrid approach:
    1.  **Core Fields:** Extract the most critical information into a predefined structure.
    2.  **Additional Data:** Capture every other piece of data on the page as generic key-value pairs.
    3.  **Line Items:** Dynamically extract all columns from the line item tables.

    **JSON Output Schema and Instructions:**

    **1. Core Information (Primary Fields):**
    - Extract these into the top level of the JSON object.
    - `document_type`: "ใบกำกับภาษี/Tax Invoice", "ใบเสร็จรับเงิน/Receipt", etc.
    - `tax_invoice_number`: The main invoice number.
    - `tax_invoice_date`: The primary date of the invoice.
    - `vendor_name`, `vendor_tax_id`, `vendor_address`: Full details of the invoice issuer.
    - `customer_name`, `customer_tax_id`, `customer_address`: Full details of the recipient.
    - `sub_total`, `vat_amount`, `grand_total`: The main financial summary.

    **2. Line Items (Comprehensive Table Extraction):**
    - `line_items`: This must be an array of objects.
    - For each invoice, identify the main table of products or services.
    - Normalize column names using the following standard keys:
        - `No.` → Item number if available
        - `Description` → Product or service name/description
        - `Quantity` → Amount or unit count
        - `Unit Price` → Price per unit
        - `Amount` → Total for that line
    - Use these exact key names even if the original table headers are in Thai or vary in wording.
    - If a row doesn't contain a value for one of these fields, include the key with a `null` value.
    
    **3. Document Checks:**
    - After extracting all data, scan the image to determine the presence of:
        - `has_tax_invoice`: true if the document contains any clear indication (text or title) that it is a tax invoice, such as the phrase "ใบกำกับภาษี".
        - `has_signature`: true if there is a visible signature or a signature-like scribble/stamp in the document.
    

    **Example JSON Output (Illustrating the Hybrid Structure):**

    ```json
    {
    "document_type": "ใบกำกับภาษี/Tax Invoice",
    "tax_invoice_number": "4104085",
    "tax_invoice_date": "04/01/25",
    "vendor_name": "RICOH SERVICES (THAILAND) LIMITED",
    "vendor_tax_id": "0105531026179",
    "vendor_address": "341 Onnuj Road, Kwaeng Prawet, Khet Prawet, Bangkok 10250",
    "customer_name": "บริษัท แมงโก้ คอนซัลแตนท์ จำกัด",
    "customer_tax_id": "0105551067687",
    "customer_address": "เลขที่ 555 อาคารรสา ทาวเวอร์ 1 ยูนิต 2304-1 ชั้นที่ 23 ถนนพหลโยธิน แขวงจตุจักร เขตจตุจักร กรุงเทพมหานคร 10900",
    "line_items": [
        {
        "รายละเอียด / Description": "ค่าเช่า/Rental Charge",
        "จำนวนเงิน / Amount": 3000.00
        }
    ],
    "sub_total": 3000.00,
    "vat_amount": 210.00,
    "grand_total": 3210.00,
    "tax_invoice": true,
    "authorized_signature": true
    }
"""

# --- PDF to image ---
def convert_pdf_to_images(pdf_bytes, dpi=300):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = [
        Image.open(BytesIO(page.get_pixmap(matrix=mat).tobytes("png")))
        for page in doc
    ]
    return images

# --- Gemini JSON extract ---
def fix_numeric_commas(json_str):
    def replace_commas_in_number(match):
        key = match.group(1)
        number = match.group(2).replace(',', '')
        return f'{key}: {number}'
    return re.sub(r'(".*?")\s*:\s*(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?)', replace_commas_in_number, json_str)

def extract_and_clean_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        json_str = fix_numeric_commas(match.group(1))
        json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

# --- Page formatting helpers ---
def format_page_ranges(pages):
    pages = sorted(set(int(p) for p in pages if pd.notnull(p)))
    if not pages:
        return ""
    ranges = []
    start = pages[0]
    end = pages[0]
    for i in range(1, len(pages)):
        if pages[i] == end + 1:
            end = pages[i]
        else:
            ranges.append(f"{start}" if start == end else f"{start}-{end}")
            start = end = pages[i]
    ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return ', '.join(ranges)

def extract_first_page_number(page_str):
    if isinstance(page_str, str):
        return int(page_str.split(',')[0].split('-')[0])
    elif isinstance(page_str, (int, float)):
        return int(page_str)
    return 0

# --- Core OCR Function ---
def run_ocr_on_pdf(pdf_bytes, start_page, end_page):
    images = convert_pdf_to_images(pdf_bytes)
    selected_images = images[start_page - 1:end_page]
    all_results = []

    for idx, img in enumerate(selected_images, start=start_page):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "contents": [{
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": "image/png", "data": b64_image}}
                ]
            }],
            "generationConfig": {"maxOutputTokens": 8192}
        }

        response = requests.post(URL, headers=HEADERS, json=payload)
        if response.ok:
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            result = extract_and_clean_json(content)
            if result:
                result['Page'] = idx
                all_results.append(result)
        time.sleep(1)

    return all_results

# --- Convert results to DataFrame ---
def results_to_dataframe(all_results):
    merged = []
    for result in all_results:
        page = result.get('Page', 0)
        items = result.pop('line_items', [])
        if isinstance(items, str):
            try:
                items = ast.literal_eval(items)
            except:
                items = []
        if items:
            for item in items:
                row = result.copy()
                row.update(item)
                row['Page'] = page
                merged.append(row)
        else:
            result['Page'] = page
            merged.append(result)

    df = pd.DataFrame(merged)
    if 'tax_invoice_number' not in df.columns:
        return pd.DataFrame()

    df_grouped = df.groupby('tax_invoice_number').agg({
        'Page': format_page_ranges,
        'document_type': 'first',
        'tax_invoice_date': 'first',
        'vendor_name': 'first',
        'vendor_tax_id': 'first',
        'vendor_address': 'first',
        'customer_name': 'first',
        'customer_tax_id': 'first',
        'customer_address': 'first',
        'sub_total': 'first',
        'vat_amount': 'first',
        'grand_total': 'first',
        'has_tax_invoice': 'first',
        'has_signature': 'last',
        'Description': lambda x: '\n'.join(x.dropna().astype(str)),
        'Quantity': lambda x: '\n'.join(x.dropna().astype(str)),
        'Unit Price': lambda x: '\n'.join(x.dropna().astype(str)),
        'Amount': lambda x: '\n'.join(x.dropna().astype(str)),
    }).reset_index()

    df_grouped['sort_page'] = df_grouped['Page'].apply(extract_first_page_number)
    df_grouped = df_grouped.sort_values(by='sort_page').drop(columns='sort_page').reset_index(drop=True)

    column_order = [
        'Page', 'document_type', 'tax_invoice_number', 'tax_invoice_date',
        'vendor_name', 'vendor_tax_id', 'vendor_address',
        'customer_name', 'customer_tax_id', 'customer_address',
        'Description', 'Quantity', 'Unit Price', 'Amount',
        'sub_total', 'vat_amount', 'grand_total',
        'has_tax_invoice', 'has_signature'
    ]

    df_grouped = df_grouped[[col for col in column_order if col in df_grouped.columns]]
    return df_grouped