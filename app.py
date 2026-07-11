from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

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

        response = model.generate_content([
            {
                "mime_type": "image/png",
                "data": image_bytes
            },
            data.question
        ])

        return {"answer": response.text.strip()}

    except Exception as e:
        return {"answer": str(e)}