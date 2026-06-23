from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from patient_model import PatientParameters
from medical_agent import MedicalAgent


app = FastAPI(
    title="Medical RAG Agent API",
    description="Cardiovascular guideline agent with ML prediction integration"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

agent = None


@app.on_event("startup")
async def startup():
    global agent
    print("🚀 Starting Medical Agent API...")
    agent = MedicalAgent()
    print("✅ API Ready!")


@app.post("/analyze")
async def analyze_patient(patient: PatientParameters):
    try:
        result = agent.analyze_patient(patient)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "online", "agent": "ready"}


@app.get("/guidelines")
async def list_guidelines():
    return {
        "guidelines": [
            "AHA_HYPERTENSION_GUIDELINES.pdf",
            "MUS_D1_chd.pdf"
        ]
    }
