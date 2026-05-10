import os
import numpy as np
import cv2
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TruthLens AI",
    description="Multimodal AI-generated content detector for images and text.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ─────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_TEXT_LENGTH = 512
IMAGE_MODEL_PATH = "models/ai_vs_human_cnn.h5"     # <-- update if needed
TEXT_MODEL_PATH  = "models/bert_text_model"         # <-- update if needed


# ── Load Models at Startup ─────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    global image_model, text_model, tokenizer

    print("Loading CNN image model...")
    image_model = load_model(IMAGE_MODEL_PATH, compile=False)
    print("CNN model loaded.")

    # print("Loading BERT text model...")
    # tokenizer   = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    # text_model  = TFAutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    # print("BERT model loaded.")


# ── Pydantic Schema for Text Request ──────────────────────────────────────────
class TextRequest(BaseModel):
    text: str


# ── Helper: allowed file type ──────────────────────────────────────────────────
def is_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Image Prediction Logic ─────────────────────────────────────────────────────
def run_image_prediction(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = float(image_model.predict(img)[0][0])

    if pred > 0.5:
        return {"label": "AI Generated",    "confidence": round(pred, 4)}
    else:
        return {"label": "Human Generated", "confidence": round(1 - pred, 4)}


# ── Text Prediction Logic ──────────────────────────────────────────────────────
def run_text_prediction(input_text: str) -> dict:
    inputs = tokenizer(
        input_text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=MAX_TEXT_LENGTH
    )

    outputs = text_model(inputs)
    probs   = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

    # Label mapping: 1 = AI Generated, 0 = Human Written
    # Flip probs[0] and probs[1] if your teammate used the reverse mapping
    ai_prob    = float(probs[1])
    human_prob = float(probs[0])

    if pred > 0.5:
        return {"label": "Human Generated", "confidence": round(pred, 4)}
    else:
        return {"label": "AI Generated",    "confidence": round(1 - pred, 4)}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not is_allowed(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, or WEBP image."
        )

    image_bytes = await file.read()
    result = run_image_prediction(image_bytes)
    result["confidence_percent"] = f"{result['confidence'] * 100:.1f}%"
    return result


# @app.post("/predict/text")
# async def predict_text(request: TextRequest):
#     text = request.text.strip()

#     if len(text) < 10:
#         raise HTTPException(
#             status_code=400,
#             detail="Text is too short. Please enter at least a sentence."
#         )

#     result = run_text_prediction(text)
#     result["confidence_percent"] = f"{result['confidence'] * 100:.1f}%"
#     return result


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)