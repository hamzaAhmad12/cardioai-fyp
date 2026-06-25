# 🫀 CardioAI — Cardiovascular Risk Assessment System

> Final Year Project · University of Peshawar · 2025

A clinical decision-support tool that combines a **Logistic Regression heart disease risk model** with a **Retrieval-Augmented Generation (RAG) pipeline** over cardiovascular clinical guidelines. The key contribution is the integrated analysis endpoint — ML predictions feed directly into the RAG prompt so recommendations are generated for the specific patient, not generically from guidelines.

---

## Screenshots

| Home | Heart Disease Prediction | Integrated Analysis |
|------|--------------------------|---------------------|
| ![Home](docs/screenshot-home.png) | ![Prediction](docs/screenshot-prediction.PNG) | ![Analysis](docs/screenshot-analysis.PNG) |

---

## Architecture

```
Streamlit UI  (frontend/app.py)
      │
      │  HTTP / REST
      ▼
FastAPI Backend  (backend/main.py)
      │
      ├── /api/ml/predict
      │       └── Logistic Regression (scikit-learn)
      │           StandardScaler → predict_proba → risk level
      │
      ├── /api/rag/chat
      │       └── ChromaDB similarity search (k=10)
      │           CrossEncoder reranker → top 4 chunks
      │           Groq Llama 3.1 8B → grounded answer + citations
      │
      └── /api/combined/analyze  ◄── key integration point
              └── ML prediction result + all 13 patient parameters
                  injected into RAG prompt before generation
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| ML model | Scikit-learn · Logistic Regression |
| RAG framework | LangChain |
| LLM | Groq · Llama 3.1 8B Instant |
| Vector database | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Language | Python 3.14 |

---

## ML Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85.7 % |
| Recall | 87.2 % |
| Precision | 84.3 % |
| F1-score | 85.7 % |
| ROC-AUC | 0.92 |

**Dataset:** UCI Heart Disease Dataset — 303 patients, cleaned to 298, 13 clinical features.  
**Algorithm:** Logistic Regression chosen over more complex models to reduce overfitting on a small medical dataset and to keep coefficients interpretable.

---

## Knowledge Base (RAG)

| Document | Topics |
|----------|--------|
| `AHA_HYPERTENSION_GUIDELINES.pdf` | Blood pressure targets, antihypertensive therapy, lifestyle modifications |
| `MUS_D1_chd.pdf` | Coronary heart disease management, angina, lipid management |

Chunks: 1 000 characters, 200-character overlap.

---

## Project Structure

```
.
├── backend/
│   ├── main.py                  # FastAPI app — all endpoints
│   ├── requirements.txt
│   ├── .env.example             # copy to .env and add GROQ_API_KEY
│   ├── models/
│   │   ├── heart_disease_logistic_model.pkl
│   │   └── heart_disease_scaler.pkl
│   ├── rag/
│   │   ├── ingest.py            # PDF → ChromaDB ingestion pipeline
│   │   ├── advanced_chat.py     # standalone terminal chatbot
│   │   ├── patient_model.py     # Pydantic patient schema
│   │   ├── agent_tools.py       # 5 LangChain tools
│   │   ├── medical_agent.py     # ReAct agent
│   │   └── data/pdfs/           # clinical guideline PDFs
│   └── chroma_db/               # generated — not committed
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
└── docs/
    ├── screenshot-home.png
    ├── screenshot-prediction.PNG
    └── screenshot-analysis.PNG
```

---

## Setup

### Prerequisites

- Python 3.10 +
- A free [Groq API key](https://console.groq.com/keys)
- Internet connection (for Groq inference)

### 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/cardioai-fyp.git
cd cardioai-fyp
```

### 2 — Backend

```bash
cd backend

# create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3 — Environment variables

```bash
cp .env.example .env
# open .env and paste your Groq API key
```

`.env` format:
```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### 4 — Build the vector database (one-time)

```bash
python rag/ingest.py
```

This reads the PDFs in `rag/data/pdfs/`, chunks them, embeds with `all-MiniLM-L6-v2`, and persists to `chroma_db/`.

### 5 — Run

**Terminal 1 — backend:**

```bash
cd backend
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive API docs)
```

**Terminal 2 — frontend:**

```bash
cd frontend
python -m streamlit run app.py
# → http://localhost:8501
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/ml/predict` | ML-only heart disease prediction |
| POST | `/api/rag/chat` | RAG chatbot — guideline-grounded Q&A |
| POST | `/api/combined/analyze` | Integrated ML + RAG personalized analysis |

Full interactive docs available at `http://localhost:8000/docs` when the backend is running.

### Example — ML prediction

```bash
curl -X POST http://localhost:8000/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 1, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 1, "thal": 2
  }'
```

```json
{
  "prediction_label": "Heart Disease",
  "risk_level": "High",
  "probability_disease": 0.7062,
  "risk_score": 0.7062
}
```

---

## Features

- **Heart Disease Prediction** — enter 13 UCI clinical parameters, get instant ML risk score (Low / Medium / High) with disease probability
- **Medical Guidelines Chatbot** — ask any cardiovascular question, get answers grounded in AHA and CHD guidelines with page-level citations
- **Integrated Analysis** — the ML prediction result and full patient profile are injected into the RAG prompt so recommendations are specific to the patient's actual risk profile, not generic summaries
- **Fully local ML** — the Logistic Regression model runs on-device with no external calls
- **Cloud LLM** — Groq provides sub-second inference on Llama 3.1 8B

---

## Team

| Name | Role | Roll No. |
|------|------|----------|
| Hamza | Developer | 157 |
| Tufail Abbas | Developer | 164 |
| Dr. Rehman Ali | Supervisor | — |

**Institution:** University of Peshawar  
**Session:** 2022–2026

---

## References

1. Janosi, A. et al. *Heart Disease Dataset.* UCI Machine Learning Repository, 1988.
2. Whelton, P. K. et al. *2017 ACC/AHA Hypertension Guidelines.* Journal of the American College of Cardiology, 2018.
3. *CHD Clinical Management Guidelines* — MUS_D1.
4. LangChain Documentation — https://docs.langchain.com
5. Groq Documentation — https://console.groq.com/docs

---

## Disclaimer

This system is developed for academic and research purposes. It is not a substitute for professional clinical judgment. Always consult a qualified healthcare professional for medical decisions.