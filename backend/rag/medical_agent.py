import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from agent_tools import (
    search_hypertension_guidelines,
    search_chd_guidelines,
    search_all_guidelines,
    analyze_patient_risk_factors,
    interpret_ml_prediction
)
from patient_model import PatientParameters

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

AGENT_PROMPT = """You are a medical guideline assistant.

Tools: {tools}
Tool names: {tool_names}

RULES:
- Call analyze_patient_risk_factors FIRST
- Call interpret_ml_prediction SECOND  
- Call ONE search tool THIRD (most relevant guideline)
- Give Final Answer IMMEDIATELY after 3 tool calls
- Never call the same tool twice
- Never call more than 3 tools total

Format strictly:
Thought: [one sentence]
Action: [tool name]
Action Input: [brief input]
Observation: [result]
Thought: [one sentence]
Action: [tool name]
Action Input: [brief input]
Observation: [result]
Thought: [one sentence]
Action: [tool name]
Action Input: [brief input]
Observation: [result]
Thought: I have enough information
Final Answer: [your complete response]

Patient Data:
{input}

{agent_scratchpad}"""


class MedicalAgent:
    def __init__(self):
        print("🔧 Initializing Medical Agent...")

        # Groq instead of Ollama
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            groq_api_key=GROQ_API_KEY,
            max_tokens=1024
        )

        self.tools = [
            search_hypertension_guidelines,
            search_chd_guidelines,
            search_all_guidelines,
            analyze_patient_risk_factors,
            interpret_ml_prediction
        ]

        prompt = PromptTemplate.from_template(AGENT_PROMPT)

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,          # increased from 8
            max_execution_time=90,      # 90 second hard timeout
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="generate"  # graceful stop → still returns output
        )

        print("✅ Medical Agent ready!")

    def _choose_guideline_tool(self, patient: PatientParameters):
        """Select the most relevant guideline search based on patient factors."""
        if patient.resting_bp_s > 140:
            return search_hypertension_guidelines
        if patient.chest_pain_type in (1, 2, 3) or patient.exercise_angina == 1 or patient.model_prediction == 1:
            return search_chd_guidelines
        return search_all_guidelines

    def _invoke_tool(self, tool, *args, **kwargs):
        """Invoke a LangChain StructuredTool using .run, .func, or direct call if available."""
        if hasattr(tool, "run") and callable(tool.run):
            return tool.run(*args, **kwargs)
        if hasattr(tool, "func") and callable(tool.func):
            return tool.func(*args, **kwargs)
        if callable(tool):
            return tool(*args, **kwargs)
        raise RuntimeError(f"Cannot invoke tool: {tool}")

    def _format_guideline_recommendation(self, guideline_result: str, patient: PatientParameters) -> str:
        """Turn retrieved guideline excerpts into concise, evidence-grounded advice."""
        prompt = f"""You are writing the recommendation section of a cardiovascular risk report.
Use ONLY the guideline excerpts below. Do not add facts, diagnoses, medicines, doses, or tests
that are not explicitly supported by those excerpts.

Write directly to the patient in clear, professional language. Do not mention retrieval, tools,
the ML pipeline, risk-factor analysis, or model interpretation. Do not repeat the source text.

If the excerpts describe unstable angina, new or worsening angina, angina at rest, or another
acute warning sign, begin with: "URGENT: seek emergency medical care now." Explain briefly why.
Otherwise provide the most appropriate next step and a short safety note to contact a heart
specialist.

If the excerpts do not support a patient-specific recommendation, reply exactly:
"I could not find a specific recommendation for this situation in the available guidelines. Please
consult a qualified heart specialist for personalized medical advice."

Patient question: {patient.patient_question or "What should I do based on my report?"}
Patient details: age {patient.age}, resting BP {patient.resting_bp_s}, chest pain type
{patient.chest_pain_type}, exercise-induced angina {patient.exercise_angina}.

Guideline excerpts:
{guideline_result}
"""
        try:
            response = self.llm.invoke(prompt).content
            return response.strip()
        except Exception as exc:
            print(f"Guideline recommendation formatting failed: {exc}")
            return (
                "I found relevant guideline information, but could not summarize it safely. "
                "Please consult a qualified heart specialist for personalized medical advice."
            )

    def analyze_patient(self, patient: PatientParameters) -> dict:
        clinical_summary = patient.to_clinical_summary()

        prediction_info = (
            f"prediction={patient.model_prediction},"
            f"probability={patient.prediction_probability},"
            f"risk={patient.risk_level()}"
        )

        query = (
            f"{patient.age}-year-old patient with resting BP {patient.resting_bp_s}, "
            f"cholesterol {patient.cholesterol}, chest pain type {patient.chest_pain_type}, "
            f"exercise angina {patient.exercise_angina}, oldpeak {patient.oldpeak}, "
            f"patient question: {patient.patient_question or 'What do my results mean?'}"
        )

        agent_input = f"""Clinical Summary:
{clinical_summary}

ML Result: {prediction_info}
Question: {patient.patient_question or "What do my results mean and what should I do?"}

Do exactly 3 steps:
1. analyze_patient_risk_factors
2. interpret_ml_prediction
3. search ONE guideline (hypertension if BP>140, CHD if chest pain, otherwise all)

Only use the tool outputs in the final response.
"""

        # Keep the ReAct execution path active for tool orchestration and logging,
        # but do not trust the model's free-form final answer.
        try:
            result = self.executor.invoke({"input": agent_input})
        except Exception as exc:
            print(f"ReAct orchestration failed; using direct tool results: {exc}")
            result = {"intermediate_steps": []}

        risk_analysis = self._invoke_tool(analyze_patient_risk_factors, clinical_summary)
        ml_analysis = self._invoke_tool(interpret_ml_prediction, prediction_info)
        guideline_tool = self._choose_guideline_tool(patient)
        guideline_result = self._invoke_tool(guideline_tool, query)

        no_guideline_info = guideline_result.lower().startswith("no relevant information found")
        if no_guideline_info:
            final_answer = (
                "I could not find relevant recommendations for this question in the available "
                "cardiovascular guidelines. Please consult a qualified heart specialist for "
                "personalized medical advice."
            )
        else:
            final_answer = self._format_guideline_recommendation(guideline_result, patient)

        tools_used = []
        for step in result.get("intermediate_steps", []):
            action, _ = step
            tools_used.append({
                "tool": action.tool,
                "input": str(action.tool_input)[:100],
            })

        if not tools_used:
            tools_used = [
                {"tool": "analyze_patient_risk_factors", "input": clinical_summary[:100]},
                {"tool": "interpret_ml_prediction", "input": prediction_info[:100]},
                {"tool": getattr(guideline_tool, "name", type(guideline_tool).__name__), "input": query[:100]},
            ]

        return {
            "answer": final_answer,
            "tools_used": tools_used,
            "risk_level": patient.risk_level(),
            "prediction": patient.model_prediction,
            "probability": patient.prediction_probability,
            "patient_age": patient.age
        }