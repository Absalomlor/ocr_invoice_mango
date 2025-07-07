from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from ocr_service import run_ocr_on_pdf
import pandas as pd

app = FastAPI(title="OCR Tax Invoice Service",)

@app.post("/ocr/json")
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


@app.post("/ocr/csv")
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
