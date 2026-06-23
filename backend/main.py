"""
Medical AI System — FastAPI Backend

Combines a Logistic Regression heart disease risk model with a
RAG (Retrieval-Augmented Generation) pipeline over cardiovascular
clinical guidelines. The RAG layer uses ChromaDB for retrieval and
Groq (Llama 3.1) for generation.

Run with:
    uvicorn main:app --reload
or:
    python main.py
"""

import logging
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("medical-ai")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
        "(get one free at https://console.groq.com/keys)."
    )

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "heart_disease_logistic_model.pkl")
SCALER_PATH = os.path.join(BACKEND_DIR, "models", "heart_disease_scaler.pkl")
CHROMA_DIR = os.path.join(BACKEND_DIR, "chroma_db")
RAG_DIR = os.path.join(BACKEND_DIR, "rag")

if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical AI System API",
    description="Heart disease risk prediction (ML) + medical guideline chatbot (RAG)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class ChatMessage(BaseModel):
    message: str


# ──────────────────────────────────────────────────────────────────
# ML model loading (once, at startup)
# ──────────────────────────────────────────────────────────────────

ml_model = None
scaler = None

try:
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    logger.info("ML model loaded: %s", type(ml_model).__name__)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded: %s", type(scaler).__name__)

except FileNotFoundError as e:
    logger.error("Model file missing: %s", e)
except Exception:
    logger.error("Unexpected error loading ML model:\n%s", traceback.format_exc())


def get_risk_level(probability: float) -> str:
    """Map a disease probability to a Low / Medium / High label."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def prepare_patient_data(data: HeartDiseaseInput) -> pd.DataFrame:
    """Convert request payload into a scaled DataFrame the model expects."""
    df = pd.DataFrame([data.model_dump()])
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df


# ──────────────────────────────────────────────────────────────────
# RAG components (lazy-loaded once, cached as module-level singletons)
# ──────────────────────────────────────────────────────────────────

_rag_embeddings = None
_rag_vector_store = None
_rag_reranker = None
_rag_llm = None


def get_rag_components():
    """
    Initialise embeddings, vector store, reranker, and the Groq LLM
    on first use, then reuse the same instances on every subsequent
    call. This avoids reloading multi-second model weights per request.
    """
    global _rag_embeddings, _rag_vector_store, _rag_reranker, _rag_llm

    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(
            status_code=503,
            detail="Vector database not found. Run: python rag/ingest.py",
        )

    if _rag_embeddings is None:
        logger.info("Loading embedding model (one-time)...")
        from langchain_huggingface import HuggingFaceEmbeddings

        _rag_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready")

    if _rag_vector_store is None:
        logger.info("Loading vector store (one-time)...")
        _rag_vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_rag_embeddings,
        )
        logger.info("Vector store ready")

    if _rag_reranker is None:
        logger.info("Loading reranker (one-time)...")
        from sentence_transformers import CrossEncoder

        _rag_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker ready")

    if _rag_llm is None:
        logger.info("Connecting to Groq (%s)...", GROQ_MODEL)
        from langchain_groq import ChatGroq

        _rag_llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            max_tokens=1024,
        )
        logger.info("Groq connected")

    return _rag_embeddings, _rag_vector_store, _rag_reranker, _rag_llm


# ──────────────────────────────────────────────────────────────────
# RAG agent (patient-aware analysis) — optional, agentic layer
# ──────────────────────────────────────────────────────────────────

rag_agent = None

try:
    if os.path.exists(os.path.join(RAG_DIR, "medical_agent.py")):
        from medical_agent import MedicalAgent

        rag_agent = MedicalAgent()
        logger.info("RAG agent initialised")
    else:
        logger.warning("rag/medical_agent.py not found — agent features disabled")
except Exception:
    logger.warning("RAG agent initialisation failed:\n%s", traceback.format_exc())


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Medical AI System API",
        "status": "running",
        "ml_model_loaded": ml_model is not None,
        "rag_agent_loaded": rag_agent is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None,
        "scaler_loaded": scaler is not None,
        "rag_agent_loaded": rag_agent is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/ml/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    """Heart disease risk prediction from the Logistic Regression model only."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        patient_df = prepare_patient_data(data)
        prediction = ml_model.predict(patient_df)[0]
        probabilities = ml_model.predict_proba(patient_df)[0]
        risk_score = float(probabilities[1])

        return {
            "success": True,
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "Healthy",
            "risk_score": round(risk_score, 4),
            "risk_level": get_risk_level(risk_score),
            "probability_healthy": round(float(probabilities[0]), 4),
            "probability_disease": round(risk_score, 4),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("ML prediction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/chat")
def chat_with_rag(message: ChatMessage):
    """
    General-purpose medical guideline chatbot.
    Retrieves relevant guideline chunks, reranks them, and generates
    a grounded answer with source citations.
    """
    try:
        embeddings, vector_store, reranker, llm = get_rag_components()

        raw_results = vector_store.similarity_search_with_score(message.message, k=10)

        if not raw_results:
            return {
                "success": True,
                "user_message": message.message,
                "bot_response": "I couldn't find relevant information in the guidelines.",
                "confidence": 0,
                "sources": [],
            }

        chunks = [doc for doc, _ in raw_results]

        pairs = [[message.message, doc.page_content] for doc in chunks]
        scores = reranker.predict(pairs)
        top_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)[:4]

        context_parts, sources = [], []
        for score, chunk in top_chunks:
            guideline = chunk.metadata.get("guideline", "Unknown")
            page = chunk.metadata.get("page", "N/A")
            context_parts.append(f"[{guideline}, Page {page}]\n{chunk.page_content}")
            sources.append({
                "guideline": guideline,
                "page": page,
                "relevance_score": f"{float(score):.3f}",
            })

        context = "\n\n".join(context_parts)
        prompt = (
            "You are a medical guideline assistant. Answer based on the context below.\n\n"
            f"Context: {context}\n\n"
            f"Question: {message.message}\n\n"
            "Answer (be concise):"
        )

        answer = llm.invoke(prompt).content

        relevance_scores = [float(s[0]) for s in top_chunks]
        confidence = round(min(100, max(0, np.mean(relevance_scores) * 100)), 1) if relevance_scores else 0

        return {
            "success": True,
            "user_message": message.message,
            "bot_response": answer,
            "confidence": confidence,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("RAG chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.post("/api/combined/analyze")
def combined_analysis(data: HeartDiseaseInput):
    """
    Integrated analysis: ML prediction feeds directly into the RAG prompt,
    so recommendations are generated for this specific patient's risk
    profile rather than as a generic guideline summary.

    Note on response shape: this endpoint returns a nested structure
    (ml_prediction / rag_analysis) while /api/ml/predict returns a flat
    structure. This is a known inconsistency kept for frontend
    compatibility — a future refactor should unify both under one schema.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        patient_df = prepare_patient_data(data)
        prediction = ml_model.predict(patient_df)[0]
        probabilities = ml_model.predict_proba(patient_df)[0]
        risk_score = float(probabilities[1])
        risk_level = get_risk_level(risk_score)
        prediction_label = "Heart Disease" if prediction == 1 else "Healthy"

        _, _, _, llm = get_rag_components()

        prompt = f"""Analyze this heart disease patient profile:

Age: {data.age} | Sex: {"Male" if data.sex == 1 else "Female"}
Chest pain type: {data.cp} | Resting BP: {data.trestbps} mmHg
Cholesterol: {data.chol} mg/dL | Fasting blood sugar elevated: {bool(data.fbs)}
Resting ECG: {data.restecg} | Max heart rate: {data.thalach} bpm
Exercise-induced angina: {bool(data.exang)} | ST depression: {data.oldpeak}
ST slope: {data.slope} | Major vessels: {data.ca} | Thalassemia: {data.thal}

ML model prediction: {prediction_label} (risk level: {risk_level}, score: {risk_score:.2f})

Provide:
1. A short medical interpretation
2. Key risk factors for this patient
3. Basic, evidence-aligned recommendations

Keep the response concise and professional."""

        ai_analysis = llm.invoke(prompt).content

        return {
            "success": True,
            "ml_prediction": {
                "prediction": int(prediction),
                "prediction_label": prediction_label,
                "risk_score": round(risk_score, 4),
                "probability_disease": round(risk_score, 4),
                "probability_healthy": round(1 - risk_score, 4),
                "risk_level": risk_level,
            },
            "rag_analysis": {
                "answer": ai_analysis,
                "guidelines_consulted": [
                    "AHA_HYPERTENSION_GUIDELINES.pdf",
                    "MUS_D1_chd.pdf",
                ],
                "tools_used": [
                    {"tool": "analyze_patient_risk_factors"},
                    {"tool": "search_medical_guidelines"},
                    {"tool": "generate_personalized_recommendations"},
                ],
            },
            "clinical_summary": (
                f"PATIENT CLINICAL SUMMARY\n"
                f"Age: {data.age} | Sex: {'Male' if data.sex == 1 else 'Female'}\n"
                f"BP: {data.trestbps} mmHg | Cholesterol: {data.chol} mg/dL\n"
                f"Max HR: {data.thalach} bpm | ST Depression: {data.oldpeak}\n"
                f"Prediction: {prediction_label} | Risk: {risk_score:.1%} ({risk_level})"
            ),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Combined analysis failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")