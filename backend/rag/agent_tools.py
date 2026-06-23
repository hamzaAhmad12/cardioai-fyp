from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


@tool
def search_hypertension_guidelines(query: str) -> str:
    """
    Search the AHA Hypertension Guidelines for blood pressure management,
    antihypertensive medications, BP targets, and lifestyle modifications.
    Use when patient has elevated BP or question is about hypertension.
    """
    store = get_vector_store()
    results = store.similarity_search(
        query,
        k=4,
        filter={"guideline": "AHA_HYPERTENSION_GUIDELINES.pdf"}
    )

    if not results:
        return "No relevant information found in AHA Hypertension Guidelines."

    output = "FROM AHA HYPERTENSION GUIDELINES:\n" + "─" * 40 + "\n"
    for doc in results:
        output += f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}\n\n"
    return output


@tool
def search_chd_guidelines(query: str) -> str:
    """
    Search the CHD Clinical Guidelines for angina, acute coronary syndromes,
    cardiac medications, and cardiovascular risk factor management.
    Use when patient has chest pain, exercise symptoms, or model predicts heart disease.
    """
    store = get_vector_store()
    results = store.similarity_search(
        query,
        k=4,
        filter={"guideline": "MUS_D1_chd.pdf"}
    )

    if not results:
        return "No relevant information found in CHD Guidelines."

    output = "FROM CHD CLINICAL GUIDELINES:\n" + "─" * 40 + "\n"
    for doc in results:
        output += f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}\n\n"
    return output


@tool
def search_all_guidelines(query: str) -> str:
    """
    Search across all available medical guidelines simultaneously.
    Use when the question spans both hypertension and heart disease,
    or when other specific tools did not return enough information.
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=6)

    if not results:
        return "No relevant information found in any guideline."

    output = "FROM ALL GUIDELINES:\n" + "\n"
    for doc in results:
        guideline = doc.metadata.get("guideline", "Unknown")
        output += f"[{guideline} | Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}\n\n"
    return output


@tool
def analyze_patient_risk_factors(clinical_summary: str) -> str:
    """
    Analyzes patient clinical parameters to identify cardiovascular risk factors.
    Use this tool first before searching any guidelines.
    Input should be the full clinical summary string.
    """
    risk_factors = []
    recommendations = []
    lines = clinical_summary.lower()

    if "elevated" in lines and "bp" in lines:
        risk_factors.append("Hypertension (Resting BP > 140 mmHg)")
        recommendations.append("Search hypertension guidelines for BP targets")

    if "high" in lines and "cholesterol" in lines:
        risk_factors.append("Hypercholesterolemia (Cholesterol > 200 mg/dL)")
        recommendations.append("Review statin therapy guidelines")

    if "typical angina" in lines:
        risk_factors.append("Typical Angina — high probability of CAD")
        recommendations.append("Search CHD guidelines for angina management")

    if "atypical angina" in lines:
        risk_factors.append("Atypical Angina — moderate probability of CAD")

    if "exercise-induced angina: yes" in lines:
        risk_factors.append("Exercise-Induced Angina — significant finding")
        recommendations.append("Search CHD guidelines for stable angina")

    if "flat" in lines or "downsloping" in lines:
        risk_factors.append("Concerning ST Slope — possible ischemia")
        recommendations.append("Search CHD guidelines for ECG findings")

    if "st-t wave abnormality" in lines or "lv hypertrophy" in lines:
        risk_factors.append("Abnormal Resting ECG")

    if "fasting blood sugar >120: yes" in lines:
        risk_factors.append("Elevated Fasting Blood Sugar — possible diabetes")
        recommendations.append("Diabetes is a major CAD risk factor")

    if "heart disease indicated" in lines:
        risk_factors.append("ML Model: Heart Disease Predicted")
        recommendations.append("Search both hypertension and CHD guidelines")

    result = "RISK FACTOR ANALYSIS:\n" +  "\n"

    if risk_factors:
        result += "Identified Risk Factors:\n"
        for rf in risk_factors:
            result += f"  ⚠️  {rf}\n"
    else:
        result += "  ✅ No major risk factors identified\n"

    if recommendations:
        result += "\nRecommended Next Steps:\n"
        for rec in recommendations:
            result += f"  → {rec}\n"

    return result


@tool
def interpret_ml_prediction(prediction_info: str) -> str:
    """
    Interprets the ML model prediction result in plain clinical language.
    Use this to explain the model output before combining with guideline info.
    Input format: "prediction=1,probability=0.82,risk=HIGH"
    """
    parts = dict(item.split("=") for item in prediction_info.split(","))
    prediction = int(parts.get("prediction", 0))
    probability = float(parts.get("probability", 0.5))
    risk = parts.get("risk", "UNKNOWN")

    if prediction == 1:
        confidence_text = (
            "very confident" if probability > 0.8
            else "moderately confident" if probability > 0.6
            else "uncertain"
        )
        action = (
            "• Urgent cardiology referral recommended" if risk == "HIGH"
            else "• Schedule cardiology consultation" if risk == "MODERATE"
            else "• Follow up with primary care physician"
        )
        return f"""
ML MODEL INTERPRETATION:
  Result: Heart Disease Indicated
  Confidence: {probability:.1%}
  Risk Level: {risk}

  The model identified patterns consistent with cardiovascular disease.
  It is {confidence_text} in this prediction.
  Clinical confirmation is required.

  Suggested Action:
  {action}
  • Further diagnostic testing may be required
  • Review and manage all identified risk factors
"""
    else:
        return f"""
ML MODEL INTERPRETATION:
  Result: No Heart Disease Indicated
  Confidence: {probability:.1%}
  Risk Level: {risk}

  The model did not identify strong patterns of cardiovascular disease.
  This does not fully rule out heart disease.
  Continue monitoring risk factors and maintain regular screening.
"""