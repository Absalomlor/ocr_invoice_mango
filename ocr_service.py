import requests
import base64
from pdf2image import convert_from_bytes
import json
import re
import time
from io import BytesIO
import pandas as pd
import ast
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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

def run_ocr_on_pdf(pdf_file: BytesIO, start_page, end_page):
    images = convert_from_bytes(pdf_file.read(), dpi=300)
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
            "generationConfig": {
                "maxOutputTokens": 8192
            }
        }

        response = requests.post(URL, headers=HEADERS, json=payload)
        if response.status_code == 200 and 'candidates' in response.json():
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            extracted = extract_and_clean_json(content)
            if extracted:
                extracted['Page'] = idx
                all_results.append(extracted)

        time.sleep(1)

    all_results_json = [r.copy() for r in all_results]

    merged_rows = []
    for result in all_results:
        page = result['Page']
        line_items = result.pop("line_items", [])
        if isinstance(line_items, str):
            try:
                line_items = ast.literal_eval(line_items)
            except:
                line_items = []

        if line_items:
            for item in line_items:
                row = result.copy()
                row.update(item)
                row['Page'] = page
                merged_rows.append(row)
        else:
            result['Page'] = page
            merged_rows.append(result)

    df = pd.DataFrame(merged_rows)
    df_grouped = df.groupby('tax_invoice_number').agg({
        'Page': 'first',
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

    df_grouped = df_grouped.sort_values(by='Page').reset_index(drop=True)

    column_order = [
        'Page',
        'document_type',
        'tax_invoice_number',
        'tax_invoice_date',
        'vendor_name',
        'vendor_tax_id',
        'vendor_address',
        'customer_name',
        'customer_tax_id',
        'customer_address',
        'Description',
        'Quantity',
        'Unit Price',
        'Amount',
        'sub_total',
        'vat_amount',
        'grand_total',
        'has_tax_invoice',
        'has_signature'
    ]

    columns_to_use = [col for col in column_order if col in df_grouped.columns]
    df_grouped = df_grouped[columns_to_use]

    return all_results_json, df_grouped