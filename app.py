import streamlit as st
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ocr_service import run_ocr_on_pdf
from PyPDF2 import PdfReader

# FastAPI App 
app = FastAPI(title="OCR Tax Invoice Service")

@app.post("/json")
async def ocr_json(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    json_data, _ = run_ocr_on_pdf(BytesIO(pdf_bytes), start_page, end_page)
    return JSONResponse(content=json_data)


@app.post("/csv")
async def ocr_csv(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    _, df = run_ocr_on_pdf(BytesIO(pdf_bytes), start_page, end_page)

    stream = BytesIO()
    df.to_csv(stream, index=False, encoding="utf-8-sig")
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ocr_result.csv"}
    )

# Streamlit App 
st.set_page_config(page_title="OCR Tax Invoice", layout="wide")

st.title("OCR ใบกำกับภาษี / Tax Invoice Extractor")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ PDF", type="pdf")

if uploaded_file:
    st.success("อัปโหลดเรียบร้อย")

    pdf_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    uploaded_file.seek(0)

    start_page = st.number_input("เริ่มหน้าที่", min_value=1, value=1, step=1)
    end_page = st.number_input("ถึงหน้าที่", min_value=start_page, value=total_pages, max_value=total_pages, step=1)

    if st.button("เริ่ม OCR"):
        with st.spinner("กำลังประมวลผล..."):
            all_results_json, df_grouped = run_ocr_on_pdf(BytesIO(pdf_bytes), start_page, end_page)

        if all_results_json:
            st.success(f"เสร็จแล้ว พบ {len(all_results_json)} หน้า")

            st.subheader("ข้อมูลจาก OCR")
            st.dataframe(df_grouped)

            csv = df_grouped.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv, file_name="ocr_result.csv", mime="text/csv")

            st.subheader("ข้อมูล JSON รายหน้า")
            for r in all_results_json:
                with st.expander(f"หน้า {r.get('Page', '?')}"):
                    st.json(r)
        else:
            st.error("ไม่พบข้อมูลที่ดึงได้")
else:
    st.info("กรุณาอัปโหลดไฟล์ PDF ก่อน")
