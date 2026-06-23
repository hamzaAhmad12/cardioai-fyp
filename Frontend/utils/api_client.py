import requests
from typing import Dict, Optional
import streamlit as st

class MedicalAPIClient:
    """
    Client for communicating with Medical AI backend
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_health(self) -> bool:
        """Check if backend is online"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def predict_heart_disease(self, patient_data: Dict) -> Optional[Dict]:
        """Get ML prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ml/predict",
                json=patient_data,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def chat_with_rag(self, message: str) -> Optional[Dict]:
        """Chat with RAG agent"""
        try:
            response = requests.post(
                f"{self.base_url}/api/rag/chat",
                json={"message": message},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            return None
    
    def combined_analysis(self, patient_data: Dict) -> Optional[Dict]:
        """Get combined ML + RAG analysis"""
        try:
            response = requests.post(
                f"{self.base_url}/api/combined/analyze",
                json=patient_data,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return None
    
    def get_statistics(self) -> Optional[Dict]:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/api/statistics", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_guidelines(self) -> Optional[Dict]:
        """Get available guidelines"""
        try:
            response = requests.get(f"{self.base_url}/api/guidelines", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None