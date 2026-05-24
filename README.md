# TruthLens AI — Multimodal Content Authenticity Detector

TruthLens is a comprehensive AI-generated content detector that combines computer vision and natural language processing to identify AI-generated images and text. The system uses a custom CNN model for image classification and a fine-tuned DistilBERT model for text analysis, providing real-time predictions through a clean web interface.

---

## Features

- 🖼️ **Image Detection** — CNN-based classifier to detect AI-generated images with high accuracy
- 📝 **Text Detection** — DistilBERT-based classifier to identify AI-generated text
- ⚡ **FastAPI Backend** — Fast, lightweight, and production-ready REST API
- 🎨 **Clean Web UI** — Minimal dark-themed frontend with instant predictions
- 🔄 **Real-time Processing** — Instant results for both text and image analysis
- 📊 **Detailed Confidence Scores** — Probability breakdown for informed decision-making

---

## Project Structure

```
C:.
│   fix_model.py
│   main.py
│   README.md
│   requirements.txt
│
├───models
│   │   ai_vs_human_cnn.h5
│   │
│   └───ai_text_detector
│           config.json
│           model.safetensors
│           tokenizer.json
│           tokenizer_config.json
│
└───static
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
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install fastapi uvicorn tensorflow opencv-python transformers torch numpy pydantic
```

### 3. Download the models

#### Image Model
The CNN model file is not included due to file size limits.

👉 Download `ai_vs_human_cnn.h5` from (https://drive.google.com/file/d/1U-6NNjyl3RuSQYQU2wwmhNBA5msatFLu/view?usp=sharing) and place it in the `models/` folder.

#### Text Model
The DistilBERT model files are not included due to file size limits.

👉 Download the `ai_text_detector` folder from (https://drive.google.com/drive/folders/16Sa7PzxuAdQOgTph6Q6y-F9OPMEUtzGy?usp=sharing) and place it in the `models/ai_text_detector` folder.

The folder should contain:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

### 4. Run the server
```bash
python main.py
```

**Or using uvicorn directly:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser at **`http://localhost:8000`**

---

## API Endpoints

### Frontend
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the web interface |

### Image Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/image` | Analyzes an uploaded image |

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@sample_image.jpg"
```

**Response:**
```json
{
  "label": "AI Generated",
  "confidence": 0.9312,
  "confidence_percent": "93.1%"
}
```

### Text Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/text` | Analyzes submitted text |

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze goes here..."}'
```

**Response:**
```json
{
  "label": "AI Generated",
  "confidence": 0.8756,
  "confidence_percent": "87.6%",
  "human_prob": 0.1244,
  "ai_prob": 0.8756
}
```

---

## Model Details

### Image Classification Model
| Property | Details |
|----------|---------|
| Architecture | Custom CNN (3 Conv blocks + Dense layers) |
| Framework | TensorFlow/Keras |
| Input Size | 224 × 224 × 3 (RGB) |
| Output | Binary classification (AI / Human) |
| Training Accuracy | ~88% |
| Dataset | [AI vs Human Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) |
| Supported Formats | PNG, JPG, JPEG, WEBP |

### Text Classification Model
| Property | Details |
|----------|---------|
| Architecture | DistilBERT (Fine-tuned) |
| Framework | PyTorch + Transformers |
| Max Sequence Length | 128 tokens |
| Output | Binary classification (Human Written / AI Generated) |
| Device Support | CUDA (GPU) / CPU auto-detection |
| Preprocessing | Lowercase, URL removal, special char filtering |

---

## Tech Stack

### Backend
- **FastAPI** — Modern, high-performance web framework
- **Uvicorn** — ASGI server for production deployment
- **Pydantic** — Data validation and settings management

### Machine Learning
- **TensorFlow/Keras** — Image model training and inference
- **PyTorch** — Text model inference
- **Transformers (HuggingFace)** — Pre-trained DistilBERT architecture
- **OpenCV** — Image preprocessing and manipulation
- **NumPy** — Numerical computations

### Frontend
- **HTML5** — Semantic markup
- **CSS3** — Modern styling with dark theme
- **Vanilla JavaScript** — No framework dependencies

---

## How It Works

### Image Detection Pipeline
1. User uploads an image (PNG, JPG, JPEG, or WEBP)
2. Image is decoded and converted to RGB
3. Resized to 224×224 pixels and normalized
4. Fed through CNN for feature extraction
5. Binary classification produces confidence score
6. Result returned: AI Generated / Human Generated

### Text Detection Pipeline
1. User submits text (minimum 10 characters)
2. Text preprocessing: lowercase, URL removal, special character filtering
3. Tokenization using DistilBERT tokenizer (max 128 tokens)
4. Model inference on GPU (if available) or CPU
5. Softmax probabilities for both classes
6. Result returned with detailed probability breakdown

---

## Configuration

### Model Paths
Update these constants in `main.py` if you change the model locations:
```python
IMAGE_MODEL_PATH = "models/ai_vs_human_cnn.h5"
TEXT_MODEL_PATH  = "models/ai_text_detector"
```

### Text Processing
```python
MAX_TEXT_LENGTH = 128  # Maximum tokens for text input
```

### Allowed Image Formats
```python
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
```

---

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload
```

### Testing Endpoints

**Test image detection:**
```python
import requests

url = "http://localhost:8000/predict/image"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Test text detection:**
```python
import requests

url = "http://localhost:8000/predict/text"
data = {"text": "This is a sample text to analyze for AI generation."}
response = requests.post(url, json=data)
print(response.json())
```

---

## Troubleshooting

### Model Loading Issues
If you encounter model loading errors:
```bash
python fix_model.py
```

### CUDA/GPU Issues
If PyTorch doesn't detect GPU:
- Verify CUDA installation: `nvidia-smi`
- Reinstall PyTorch with CUDA support
- The system will automatically fall back to CPU

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

---

## Future Enhancements

- [ ] Batch processing for multiple files
- [ ] Drag-and-drop file upload interface
- [ ] Confidence threshold customization
- [ ] Export detection reports (PDF/CSV)
- [ ] Audio deepfake detection
- [ ] Video content analysis
- [ ] REST API authentication
- [ ] Model performance monitoring and drift detection

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Dataset**: [AI vs Human Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)
- **DistilBERT**: HuggingFace Transformers library
- **Framework**: FastAPI team for excellent documentation

---

## Authors

**Zuhair Ahmed**  
**Rumesa Iqbal**

Built as part of DLP (Deep Learning Project)  
FAST NUCES — Semester 6

---

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Submit a pull request
- Contact the development team

---

**⚡ TruthLens AI — Empowering authenticity in the age of synthetic content**
