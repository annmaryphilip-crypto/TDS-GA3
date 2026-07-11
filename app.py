from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import os
import re
from datetime import datetime

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="GA3 Image QA API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def home():
    return {"message": "GA3 API is running"}


@app.post("/answer-image")
async def answer_image(data: ImageRequest):
    try:
        image_bytes = base64.b64decode(data.image_base64)

        prompt = f"""
You are an OCR and visual question answering assistant.

Answer the following question about the image.

Question:
{data.question}

Rules:
- Return ONLY the answer.
- Do NOT explain.
- Do NOT include units unless explicitly asked.
- For numbers return only the number.
- For text return only the exact text.
"""

        response = model.generate_content(
            [
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_bytes,
                },
            ]
        )

        answer = response.text.strip()

        # Remove markdown formatting if Gemini adds it
        answer = answer.replace("```", "").replace("`", "").strip()

        return {"answer": str(answer)}

    except Exception as e:
        return {"answer": str(e)}
    


class InvoiceRequest(BaseModel):
    invoice_text: str

def extract(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def parse_amount(value):
    if value is None:
        return None
    value = value.replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    return float(m.group(1)) if m else None

def parse_date(value):
    if value is None:
        return None

    value = value.strip()

    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%d/%m/%Y"
    ]

    for f in formats:
        try:
            return datetime.strptime(value, f).strftime("%Y-%m-%d")
        except:
            pass

    return None


@app.post("/extract")
async def extract_invoice(data: InvoiceRequest):

    prompt = f"""
You are an invoice extraction system.

Extract the following fields from the invoice.

Return ONLY valid JSON.

Fields:

invoice_no
date
vendor
amount
tax
currency

Rules:

- invoice_no = invoice number
- date = ISO format YYYY-MM-DD
- vendor = seller/vendor/company name
- amount = subtotal BEFORE tax
- tax = tax amount ONLY (NOT percentage)
- currency = INR, USD, EUR etc.
- If a field is missing use null.
- Return ONLY JSON.
- Do NOT explain anything.

Invoice:

{data.invoice_text}
"""

    try:

        response = model.generate_content(prompt)

        text = response.text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        import json

        result = json.loads(text)

        return {
            "invoice_no": result.get("invoice_no"),
            "date": result.get("date"),
            "vendor": result.get("vendor"),
            "amount": result.get("amount"),
            "tax": result.get("tax"),
            "currency": result.get("currency")
        }

    except Exception:

        return {
            "invoice_no": None,
            "date": None,
            "vendor": None,
            "amount": None,
            "tax": None,
            "currency": None
        }