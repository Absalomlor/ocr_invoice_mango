from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ocr_service import run_ocr_on_pdf, results_to_dataframe
from io import BytesIO
import pandas as pd

app = FastAPI()

@app.post("/ocr/json")
async def ocr_json(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    pdf_bytes = await file.read()
    results = run_ocr_on_pdf(pdf_bytes, start_page, end_page)
    return JSONResponse(content={"results": results})

@app.post("/ocr/csv")
async def ocr_csv(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    pdf_bytes = await file.read()
    results = run_ocr_on_pdf(pdf_bytes, start_page, end_page)
    df = results_to_dataframe(results)

    stream = BytesIO()
    df.to_csv(stream, index=False, encoding="utf-8-sig")
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ocr_results.csv"}
    )
