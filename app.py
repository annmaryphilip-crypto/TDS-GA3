from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import os

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