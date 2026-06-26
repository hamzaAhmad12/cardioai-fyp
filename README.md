# CardioAI — Cardiovascular Risk Assessment System

> Final Year Project · University of Peshawar · 2025–2026

A clinical decision-support system combining a **Logistic Regression heart disease risk model** with a **Retrieval-Augmented Generation (RAG) pipeline** over five cardiovascular clinical guidelines. The key contribution is the integrated analysis endpoint — ML predictions and patient parameters feed directly into the RAG prompt so recommendations are generated for the specific patient, not generically from guidelines.

---

## Screenshots

| Home | Heart Disease Prediction | Integrated Analysis |
|------|--------------------------|---------------------|
| ![Home](docs/screenshot-home.png) | ![Prediction](docs/screenshot-prediction.PNG) | ![Analysis](docs/screenshot-analysis.PNG) |

---

## Architecture

```
Frontend/app.py  (Streamlit)
      │
      │  HTTP / REST
      ▼
backend/main.py  (FastAPI)
      │
      ├── /api/ml/predict
      │       └── Logistic Regression · StandardScaler · predict_proba
      │
      ├── /api/rag/chat
      │       └── ChromaDB similarity search (k=10)
      │           CrossEncoder reranker → top 4 chunks
      │           Groq Llama 3.1 8B → grounded answer + citations
      │
      └── /api/combined/analyze   ← key integration point
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
| LLM | Groq · Llama 3.1 8B Instant (cloud) |
| Vector database | ChromaDB (local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
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
**Algorithm:** Logistic Regression chosen for interpretability and reduced overfitting on a small medical dataset.

---

## Knowledge Base (RAG)

| Document | Topics |
|----------|--------|
| `AHA HYPERTENSION GUIDELINES.pdf` | Blood pressure targets, antihypertensive therapy, lifestyle modifications |
| `MUS_D1_chd.pdf` | Coronary heart disease management, angina, lipid management |
| `ESC GUIDELINES.pdf` | European cardiovascular prevention guidelines |
| `NCEP ATP III cholesterol guidelines.pdf` | Cholesterol management, LDL targets, statin therapy |
| `WHO CARDIOVASCULAR GUIDELINES.pdf` | Global cardiovascular risk assessment framework |

Chunking: 1 000 characters per chunk, 200-character overlap.

---

## Project Structure

```
.
├── Frontend/
│   ├── app.py                   # Streamlit UI — all pages in one file
│   └── requirements.txt
│
├── backend/
│   ├── main.py                  # FastAPI app — all endpoints
│   ├── requirements.txt
│   ├── .env.example             # copy to .env and add your GROQ_API_KEY
│   ├── start_backend.bat        # Windows quick-start script
│   ├── models/
│   │   ├── heart_disease_logistic_model.pkl   # primary model
│   │   ├── heart_disease_scaler.pkl           # StandardScaler
│   │   └── heart_disease_model_cleaned.pkl    # RandomForest backup
│   └── rag/
│       ├── ingest.py            # PDF → ChromaDB ingestion (run once)
│       ├── patient_model.py     # Pydantic patient schema
│       ├── agent_tools.py       # 5 LangChain tools
│       ├── medical_agent.py     # ReAct agent
│       └── data/pdfs/           # 5 clinical guideline PDFs
│
├── Data/
│   └── heart_cleaned.csv        # cleaned UCI dataset used for training
│
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)
- Internet connection (Groq inference is cloud-based)

### 1 — Clone

```bash
git clone https://github.com/hamzaAhmad12/cardioai-fyp.git
cd cardioai-fyp
```

### 2 — Backend

```bash
cd backend

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3 — Environment variables

```bash
copy .env.example .env
# open .env and paste your Groq API key
```

`.env` format:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### 4 — Build vector database (one-time)

```bash
python rag/ingest.py
```

Reads all PDFs in `rag/data/pdfs/`, chunks them, embeds with `all-MiniLM-L6-v2`, and persists to `chroma_db/`.

### 5 — Run

**Terminal 1 — backend:**

```bash
cd backend
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Terminal 2 — frontend:**

```bash
cd Frontend
python -m streamlit run app.py
# UI: http://localhost:8501
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/ml/predict` | ML-only heart disease prediction |
| POST | `/api/rag/chat` | RAG chatbot — guideline-grounded Q&A |
| POST | `/api/combined/analyze` | Integrated ML + RAG personalised analysis |

Interactive docs at `http://localhost:8000/docs` when the backend is running.

### Example request

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
  "probability_disease": 0.7062
}
```

---

## Key Integration

The `/api/combined/analyze` endpoint passes the ML prediction result **and** all 13 patient parameters into the RAG prompt before generation. This means the LLM produces recommendations specific to the patient's actual risk profile — not generic guideline summaries. This was the core feedback from our supervisor and the main technical contribution of the project.

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
2. Whelton, P. K. et al. *2017 ACC/AHA Hypertension Guidelines.* JACC, 2018.
3. *ESC Guidelines on Cardiovascular Prevention,* 2021.
4. *NCEP ATP III Guidelines on Cholesterol Management,* 2002.
5. *WHO Cardiovascular Disease Prevention Guidelines,* 2007.
6. *CHD Clinical Management Guidelines* — MUS_D1.
7. LangChain Documentation — https://docs.langchain.com
8. Groq Documentation — https://console.groq.com/docs

---

## Disclaimer

This system is developed for academic and research purposes only. It does not replace professional clinical judgment. Always consult a qualified healthcare professional for medical decisions.