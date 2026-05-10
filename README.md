# TruthLens AI — Content Authenticity Detector

TruthLens is a multimodal AI-generated content detector that uses a CNN model for images and a BERT model for text to classify content as AI-generated or human-created. Built with FastAPI and TensorFlow, it provides a clean web interface for real-time predictions.

---

## Features

- 🖼️ **Image Detection** — CNN-based classifier to detect AI-generated images
- 📝 **Text Detection** — BERT-based classifier to detect AI-generated text *(coming soon)*
- ⚡ **FastAPI Backend** — Fast and lightweight REST API
- 🎨 **Clean Web UI** — Minimal dark-themed frontend, no frameworks needed

---

## Project Structure

```
TruthLens/
│   main.py
│   fix_model.py
│
├───models/
│       ai_vs_human_cnn.h5        # Download separately (see below)
│
└───static/
        index.html
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/truthlens.git
cd truthlens
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn tensorflow opencv-python transformers
```

### 3. Download the model
The `.h5` model file is not included in this repo due to file size limits.

👉 Download `ai_vs_human_cnn.h5` from [https://drive.google.com/file/d/1U-6NNjyl3RuSQYQU2wwmhNBA5msatFLu/view?usp=sharing](#) and place it inside the `models/` folder.

### 4. Run the server
```bash
uvicorn main:app --reload
```

Then open your browser at `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the frontend |
| POST | `/predict/image` | Predicts if an image is AI or human generated |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@your_image.jpg"
```

### Example Response
```json
{
  "label": "AI Generated",
  "confidence": 0.9312,
  "confidence_percent": "93.1%"
}
```

---

## Model Details

| Property | Details |
|----------|---------|
| Architecture | Custom CNN (3 Conv blocks) |
| Input Size | 224 × 224 × 3 |
| Output | Binary (AI / Human) |
| Training Accuracy | ~88% |
| Dataset | [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) |

---

## Tech Stack

- **Backend** — FastAPI, Uvicorn
- **ML** — TensorFlow, Keras, OpenCV
- **Frontend** — HTML, CSS, Vanilla JS
- **NLP** *(planned)* — HuggingFace Transformers, BERT

---

## Future Work

- [ ] Enable BERT text detection endpoint
- [ ] Add drag and drop image upload
- [ ] Deploy on Hugging Face Spaces or Render

---

## Authors

Built as part of DLP Project — FAST NUCES Semester 6
