from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from agent_tools import (
    search_hypertension_guidelines,
    search_chd_guidelines,
    search_all_guidelines,
    analyze_patient_risk_factors,
    interpret_ml_prediction
)
from patient_model import PatientParameters


AGENT_PROMPT = """You are an expert medical guideline assistant integrated 
with a cardiovascular disease prediction system.

You receive patient clinical parameters, an ML model prediction, 
and an optional patient question.

Your job:
1. Analyze the patient risk factors using analyze_patient_risk_factors
2. Interpret the ML prediction using interpret_ml_prediction
3. Search relevant guidelines based on the patient situation
4. Synthesize a comprehensive, grounded response

Tool selection rules:
- Resting BP > 140: search hypertension guidelines
- Chest pain type 1 or 2: search CHD guidelines
- Exercise angina = Yes: search CHD guidelines
- Model prediction = 1: search both guidelines
- Cholesterol > 240: search CHD guidelines
- ST slope flat or downsloping: search CHD guidelines
- General questions: search all guidelines

Safety rules:
- Always remind patient this is informational only
- Always recommend consulting a doctor
- Never prescribe specific medications

Available tools:
{tools}

Tool names: {tool_names}

Format:
Thought: your reasoning
Action: tool name
Action Input: input to tool
Observation: tool result
... repeat as needed ...
Thought: I now have enough information
Final Answer: comprehensive response to patient

Begin!

Patient Data:
{input}

{agent_scratchpad}"""


class MedicalAgent:
    def __init__(self):
        print("🔧 Initializing Medical Agent...")

        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.1,
            num_ctx=4096
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
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        print("✅ Medical Agent ready!\n")

    def analyze_patient(self, patient: PatientParameters) -> dict:
        clinical_summary = patient.to_clinical_summary()

        prediction_info = (
            f"prediction={patient.model_prediction},"
            f"probability={patient.prediction_probability},"
            f"risk={patient.risk_level()}"
        )

        agent_input = f"""
{clinical_summary}

Prediction Info (for interpret_ml_prediction tool): {prediction_info}

Patient Question: {patient.patient_question or 
                   'Please explain my results and what they mean.'}

Analyze this patient cardiovascular risk based on their parameters 
and ML model prediction. Search the relevant medical guidelines and 
provide a comprehensive grounded response.
"""

        print()
        print(f"🤖 Agent Processing Patient "
              f"(Age {patient.age}, Risk: {patient.risk_level()})")
        print()

        result = self.executor.invoke({"input": agent_input})

        tools_used = []
        for step in result.get("intermediate_steps", []):
            action, _ = step
            tools_used.append({
                "tool": action.tool,
                "input": str(action.tool_input)[:100],
            })

        return {
            "answer": result["output"],
            "tools_used": tools_used,
            "risk_level": patient.risk_level(),
            "prediction": patient.model_prediction,
            "probability": patient.prediction_probability,
            "patient_age": patient.age
        }