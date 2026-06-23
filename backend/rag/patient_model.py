# patient_model.py
# Defines the data structure coming from your Flutter UI

from pydantic import BaseModel, Field
from typing import Optional

class PatientParameters(BaseModel):
    """
    Matches exactly the columns your ML model was trained on.
    Flutter UI sends this as a JSON POST request.
    """
    # Demographics
    age: int = Field(..., ge=1, le=120, description="Patient age")
    sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    
    # Cardiac Parameters
    chest_pain_type: int = Field(
        ..., ge=1, le=4,
        description="1=Typical Angina, 2=Atypical Angina, "
                    "3=Non-anginal Pain, 4=Asymptomatic"
    )
    resting_bp_s: float = Field(
        ..., ge=0, description="Resting systolic blood pressure (mmHg)"
    )
    cholesterol: float = Field(
        ..., ge=0, description="Serum cholesterol (mg/dL)"
    )
    fasting_blood_sugar: int = Field(
        ..., ge=0, le=1,
        description="1 if fasting BS > 120 mg/dL, else 0"
    )
    resting_ecg: int = Field(
        ..., ge=0, le=2,
        description="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"
    )
    max_heart_rate: float = Field(
        ..., ge=0, description="Maximum heart rate achieved"
    )
    exercise_angina: int = Field(
        ..., ge=0, le=1,
        description="1=Yes (exercise-induced angina), 0=No"
    )
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise"
    )
    st_slope: int = Field(
        ..., ge=1, le=3,
        description="1=Upsloping, 2=Flat, 3=Downsloping"
    )
    
    # ML Model Output (from your pre-trained model)
    model_prediction: int = Field(
        ..., ge=0, le=1,
        description="0=No heart disease, 1=Heart disease predicted"
    )
    prediction_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence of the prediction (0.0 to 1.0)"
    )
    
    # Optional question from patient
    patient_question: Optional[str] = Field(
        None,
        description="Optional question typed by patient in Flutter UI"
    )

    def risk_level(self) -> str:
        """Quick risk tier based on prediction probability."""
        prob = self.prediction_probability
        if prob >= 0.75:
            return "HIGH"
        elif prob >= 0.50:
            return "MODERATE"
        elif prob >= 0.25:
            return "LOW-MODERATE"
        else:
            return "LOW"
    
    def to_clinical_summary(self) -> str:
        """
        Converts raw numbers into a readable clinical summary
        for the LLM agent to understand.
        """
        sex_str = "Male" if self.sex == 1 else "Female"
        
        chest_pain_map = {
            1: "Typical Angina",
            2: "Atypical Angina", 
            3: "Non-anginal Pain",
            4: "Asymptomatic"
        }
        
        ecg_map = {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }
        
        slope_map = {
            1: "Upsloping (generally favorable)",
            2: "Flat (concerning)",
            3: "Downsloping (most concerning)"
        }
        
        # Flag abnormal values
        bp_flag = "⚠️ ELEVATED" if self.resting_bp_s > 140 else "✅ Normal"
        chol_flag = "⚠️ HIGH" if self.cholesterol > 200 else "✅ Normal"
        hr_flag = ("⚠️ LOW" if self.max_heart_rate < 100 
                   else "✅ Adequate")
        
        return f"""
PATIENT CLINICAL SUMMARY

Demographics:
  • Age: {self.age} years
  • Sex: {sex_str}

Cardiac Parameters:
  • Chest Pain Type: {chest_pain_map.get(self.chest_pain_type, 'Unknown')}
  • Resting BP: {self.resting_bp_s} mmHg [{bp_flag}]
  • Cholesterol: {self.cholesterol} mg/dL [{chol_flag}]
  • Fasting Blood Sugar >120: {'Yes ⚠️' if self.fasting_blood_sugar else 'No ✅'}
  • Resting ECG: {ecg_map.get(self.resting_ecg, 'Unknown')}
  • Max Heart Rate: {self.max_heart_rate} bpm [{hr_flag}]
  • Exercise-Induced Angina: {'Yes ⚠️' if self.exercise_angina else 'No ✅'}
  • ST Depression (Oldpeak): {self.oldpeak}
  • ST Slope: {slope_map.get(self.st_slope, 'Unknown')}

ML Model Assessment:
  • Prediction: {'⚠️ HEART DISEASE INDICATED' if self.model_prediction == 1 
                  else '✅ NO HEART DISEASE INDICATED'}
  • Confidence: {self.prediction_probability:.1%}
  • Risk Level: {self.risk_level()}
"""