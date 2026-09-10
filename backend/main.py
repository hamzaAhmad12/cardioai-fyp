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
from typing import Optional

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
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
        "(get one free at https://console.groq.com/keys)."
    )

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "heart_disease_logistic_model.pkl")
SCALER_PATH = os.path.join(BACKEND_DIR, "models", "heart_disease_scaler.pkl")
ENCODER_PATH = os.path.join(BACKEND_DIR, "models", "heart_disease_encoder.pkl")  # NEW
CHROMA_DIR = os.path.join(BACKEND_DIR, "chroma_db")
RAG_DIR = os.path.join(BACKEND_DIR, "rag")

if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}

# Categorical columns that were one-hot encoded during training
CAT_COLS = ["cp", "thal", "restecg", "slope", "ca"]

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
# ML model + scaler + encoder loading (once, at startup)
# ──────────────────────────────────────────────────────────────────

ml_model = None
scaler = None
encoder = None          # NEW

try:
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    logger.info("ML model loaded: %s", type(ml_model).__name__)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded: %s", type(scaler).__name__)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    logger.info("OneHotEncoder loaded: %s", type(encoder).__name__)

except FileNotFoundError as e:
    logger.error("Model / scaler / encoder file missing: %s", e)
except Exception:
    logger.error("Unexpected error loading ML artifacts:\n%s", traceback.format_exc())


def get_risk_level(probability: float) -> str:
    """Map a disease probability to a Low / Medium / High label."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def prepare_patient_data(data: HeartDiseaseInput) -> pd.DataFrame:
    """
    Convert the original 13 features → OneHotEncoder (same as training)
    → StandardScaler → DataFrame ready for the model.
    """
    if encoder is None or scaler is None:
        raise RuntimeError("Encoder or Scaler not loaded")

    # 1. Create DataFrame with the exact original columns
    df = pd.DataFrame([data.model_dump()])

    # 2. Apply the same OneHotEncoder that was used in training
    encoded_array = encoder.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=df.index,
    )

    # 3. Keep the numeric columns in the same order as training
    #    (drop the original categorical columns)
    numeric_df = df.drop(columns=CAT_COLS)

    # 4. Concatenate → this produces the exact same column structure
    #    that the scaler and model saw during training
    X = pd.concat([numeric_df, encoded_df], axis=1)

    # 5. Scale
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df


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
# RAG agent (patient-aware analysis)
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
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None,          # NEW
        "rag_agent_loaded": rag_agent is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None,          # NEW
        "rag_agent_loaded": rag_agent is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/ml/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    """Heart disease risk prediction from the Logistic Regression model only."""
    if ml_model is None or encoder is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model / encoder / scaler not loaded")

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


# ============================================
# COMBINED ANALYSIS ENDPOINT — REAL INTEGRATION
# ML prediction → PatientParameters → ReAct Agent → Real RAG
# ============================================

class CombinedAnalysisInput(BaseModel):
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
    patient_question: Optional[str] = None

@app.post("/api/combined/analyze")
def combined_analysis(data: CombinedAnalysisInput):
    """
    TRUE INTEGRATED ANALYSIS:
    1. Run ML model on patient parameters
    2. Create PatientParameters object (from patient_model.py)
    3. Pass to MedicalAgent.analyze_patient() — the REAL ReAct agent
    4. Agent selects real tools, runs real ChromaDB retrieval, real reranking
    5. Returns actual tools used, actual guideline chunks retrieved
    """
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Agent not initialized. Make sure Ollama/Groq is running."
        )
    
    try:
        import traceback
        
        print("\n" + "="*60)
        print(" REAL COMBINED ANALYSIS STARTED")
        print("="*60)
        
        # ────────────────────────────────────────────────────
        # STEP 1: Run ML Prediction
        # ────────────────────────────────────────────────────
        print("\n Step 1: Running ML Prediction...")
        
        heart_input = HeartDiseaseInput(
            age=data.age, sex=data.sex, cp=data.cp,
            trestbps=data.trestbps, chol=data.chol,
            fbs=data.fbs, restecg=data.restecg,
            thalach=data.thalach, exang=data.exang,
            oldpeak=data.oldpeak, slope=data.slope,
            ca=data.ca, thal=data.thal
        )
        
        patient_df = prepare_patient_data(heart_input)
        prediction = ml_model.predict(patient_df)[0]
        probabilities = ml_model.predict_proba(patient_df)[0]
        risk_score = float(probabilities[1])
        risk_level = get_risk_level(risk_score)
        prediction_label = "Heart Disease Detected" if prediction == 1 else "Healthy"
        
        logger.info(f"✅ ML Prediction: {prediction_label} | Risk: {risk_level} ({risk_score:.2%})")
        
        # ────────────────────────────────────────────────────
        # STEP 2: Build PatientParameters for the Real Agent
        # ────────────────────────────────────────────────────
        print("\n Step 2: Building PatientParameters object...")
        
        from patient_model import PatientParameters
        
        patient_params = PatientParameters(
            age=data.age,
            sex=data.sex,
            chest_pain_type=data.cp,
            resting_bp_s=data.trestbps,
            cholesterol=data.chol,
            fasting_blood_sugar=data.fbs,
            resting_ecg=data.restecg,
            max_heart_rate=data.thalach,
            exercise_angina=data.exang,
            oldpeak=data.oldpeak,
            st_slope=data.slope,
            model_prediction=int(prediction),
            prediction_probability=risk_score,
            patient_question=data.patient_question or "What do my results mean and what should I do?"
        )
        
        logger.info(f"✅ PatientParameters built | Risk Level: {patient_params.risk_level()}")
        
        # ────────────────────────────────────────────────────
        # STEP 3: Run the REAL ReAct Agent
        # This calls the actual agent with real tools:
        # - analyze_patient_risk_factors (real)
        # - interpret_ml_prediction (real)
        # - search_hypertension_guidelines (real ChromaDB)
        # - search_chd_guidelines (real ChromaDB)
        # - search_all_guidelines (real ChromaDB)
        # ────────────────────────────────────────────────────
        print("\n Step 3: Running REAL ReAct Agent...")
        print(f"   Patient question: {patient_params.patient_question}")
        
        rag_result = rag_agent.analyze_patient(patient_params)
        
        actual_tools_used = rag_result.get("tools_used", [])
        logger.info(f"✅ Agent completed | Tools actually used: {len(actual_tools_used)}")
        
        for tool in actual_tools_used:
            logger.info(f"   - {tool.get('tool', 'unknown')}")
        
        # ────────────────────────────────────────────────────
        # STEP 4: Build Clinical Summary
        # ────────────────────────────────────────────────────
        clinical_summary = patient_params.to_clinical_summary()
        
        # ────────────────────────────────────────────────────
        # STEP 5: Return REAL Results
        # ────────────────────────────────────────────────────
        print("\n✅ REAL COMBINED ANALYSIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            "success": True,
            
            # Real ML prediction
            "ml_prediction": {
                "prediction": int(prediction),
                "prediction_label": prediction_label,
                "risk_score": round(risk_score, 4),
                "probability_disease": round(risk_score, 4),
                "probability_healthy": round(float(probabilities[0]), 4),
                "risk_level": risk_level,
            },
            
            # Real RAG agent results (actual tools used, actual retrieved content)
            "rag_analysis": {
                "answer": rag_result.get("answer", "No answer generated"),
                "tools_used": actual_tools_used,  # REAL tools, not hardcoded
                "guidelines_consulted": list(set(
                    tool.get("tool", "") 
                    for tool in actual_tools_used
                    if "guideline" in tool.get("tool", "").lower() 
                    or "search" in tool.get("tool", "").lower()
                )),
            },
            
            # Patient clinical summary
            "clinical_summary": clinical_summary,
            
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Combined analysis failed:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Combined analysis error: {str(e)}")


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")