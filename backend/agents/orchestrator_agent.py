import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from . import diagnosis_agent, treatment_agent, monitoring_agent, report_agent
from .specialist_agents import (
    CardiologyAgent, NeurologyAgent, PharmacologyAgent,
    PsychiatryAgent, PulmonologyAgent, GastroenterologyAgent,
    DermatologyAgent, EndocrinologyAgent
)

class PatientContext:
    def __init__(self):
        self.conversation_history = []
        self.patient_profile = {}
        self.current_symptoms = ""
        self.current_diagnosis = ""
        self.current_treatment = ""
        self.specialist_consultations = {}

    def add_interaction(self, interaction_type: str, content: str, agent: str):
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "content": content,
            "agent": agent
        })

    def get_context_summary(self) -> str:
        return f"""
            Patient Context:
            - Current Symptoms: {self.current_symptoms}
            - Working Diagnosis: {self.current_diagnosis}
            - Treatment Plan: {self.current_treatment}
            - Previous Consultations: {len(self.conversation_history)} interactions
            """


class MedicalOrchestrator:
    def __init__(self, model):
        self.model = model
        self.context = PatientContext()
        self.available_agents = {
            "diagnosis": diagnosis_agent,
            "treatment": treatment_agent,
            "monitoring": monitoring_agent,
            "report": report_agent
        }
        self.specialist_agents = {
            "cardiology": CardiologyAgent(model),
            "neurology": NeurologyAgent(model),
            "pharmacology": PharmacologyAgent(model),
            "psychiatry": PsychiatryAgent(model),
            "pulmonology": PulmonologyAgent(model),
            "gastroenterology": GastroenterologyAgent(model),
            "dermatology": DermatologyAgent(model),
            "endocrinology": EndocrinologyAgent(model)
        }

    def analyze_query_intent(self, user_input: str) -> Dict[str, any]:
        prompt = f"""
            You are a medical AI coordinator. Analyze the following user input and determine:

            1. Primary intent (diagnosis, treatment, monitoring, reporting, emergency)
            2. Urgency level (low, medium, high, emergency)
            3. Required specialist consultation (general, cardiology, neurology, pharmacology, etc.)
            4. Patient data needed (symptoms, history, vitals, medications)

            User Input: "{user_input}"

            Respond in JSON format:
            {{
                "intent": "diagnosis|treatment|monitoring|reporting|emergency",
                "urgency": "low|medium|high|emergency", 
                "specialists": ["general", "cardiology", "neurology", "pharmacology"],
                "data_needed": ["symptoms", "history", "vitals", "medications"],
                "workflow_steps": ["step1", "step2", "step3"]
            }}
            """
        try:
            response = self.model.generate_response(prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response[json_start:json_end])
        except:
            pass

        user_lower = user_input.lower()
        if any(word in user_lower for word in ['pain', 'hurt', 'ache', 'symptom']):
            return {
                "intent": "diagnosis",
                "urgency": "medium",
                "specialists": ["general"],
                "data_needed": ["symptoms"],
                "workflow_steps": ["validate_input", "diagnose", "recommend_treatment"]
            }
        return {
            "intent": "general",
            "urgency": "low",
            "specialists": ["general"],
            "data_needed": ["symptoms"],
            "workflow_steps": ["validate_input", "diagnose"]
        }

    def coordinate_diagnosis_workflow(self, symptoms: str) -> Tuple[str, str, Dict]:
        workflow_log = {
            "steps": [],
            "agents_consulted": [],
            "confidence_scores": {},
            "recommendations": []
        }

        workflow_log["steps"].append("Initial Triage")
        if not diagnosis_agent.is_input_medical(symptoms, self.model):
            return "❌ Input not medically relevant", "", workflow_log

        workflow_log["steps"].append("Primary Diagnosis")
        workflow_log["agents_consulted"].append("diagnosis_agent")

        primary_diagnosis = diagnosis_agent.generate_diagnosis(symptoms)
        self.context.current_symptoms = symptoms
        self.context.current_diagnosis = primary_diagnosis

        specialist = self.determine_specialist_consultation(symptoms, primary_diagnosis)
        if specialist and specialist in self.specialist_agents:
            workflow_log["steps"].append(f"Specialist Consultation: {specialist}")
            workflow_log["agents_consulted"].append(specialist)
            specialist_diagnosis = self.specialist_agents[specialist].model.generate_response(symptoms)
            primary_diagnosis = f"{primary_diagnosis}\n\n**Specialist Opinion ({specialist}):**\n{specialist_diagnosis}"

        workflow_log["steps"].append("Treatment Planning")
        workflow_log["agents_consulted"].append("treatment_agent")

        treatment_plan = treatment_agent.generate_treatment(symptoms, primary_diagnosis, self.model)
        self.context.current_treatment = treatment_plan

        workflow_log["steps"].append("Safety Validation")
        safety_check = self.validate_treatment_safety(symptoms, primary_diagnosis, treatment_plan)
        if not safety_check["safe"]:
            treatment_plan = f"⚠️ **SAFETY ALERT**: {safety_check['warning']}\n\n{treatment_plan}"

        self.context.add_interaction("diagnosis", primary_diagnosis, "diagnosis_agent")
        self.context.add_interaction("treatment", treatment_plan, "treatment_agent")

        workflow_log["recommendations"] = [
            "Consider follow-up in 24-48 hours",
            "Monitor for symptom changes", 
            "Seek immediate care if symptoms worsen"
        ]

        return primary_diagnosis, treatment_plan, workflow_log

    def coordinate_monitoring_workflow(self, patient_id: str) -> Tuple[str, Optional[str]]:
        """Run monitoring agent on existing patient history."""
        summary, chart_path = monitoring_agent.analyze_patient_history(patient_id)
        self.context.add_interaction("monitoring", summary, "monitoring_agent")
        return summary, chart_path

    def determine_specialist_consultation(self, symptoms: str, diagnosis: str) -> Optional[str]:
        specialist_keywords = {
            "cardiology": ["chest", "heart", "palpitations"],
            "neurology": ["headache", "seizure", "numbness", "dizziness"],
            "pharmacology": ["drug", "medication", "dose", "interaction"],
            "psychiatry": ["depression", "mood", "anxiety", "suicidal", "sleep"],
            "pulmonology": ["cough", "asthma", "breath", "wheezing"],
            "gastroenterology": ["stomach", "nausea", "vomit", "digestion"],
            "dermatology": ["rash", "skin", "itch"],
            "endocrinology": ["diabetes", "thyroid", "hormone"]
        }
        lower_symptoms = symptoms.lower()
        for specialty, keywords in specialist_keywords.items():
            if any(kw in lower_symptoms for kw in keywords):
                return specialty
        return None

    def validate_treatment_safety(self, symptoms: str, diagnosis: str, treatment: str) -> Dict[str, any]:
        prompt = f"""
                You are a medical safety validator. Analyze this treatment plan for potential safety concerns:

                Symptoms: {symptoms}
                Diagnosis: {diagnosis}  
                Treatment: {treatment}

                Check for:
                1. Dangerous drug interactions
                2. Contraindications
                3. Dosing errors
                4. Missing critical warnings
                5. Emergency symptoms that need immediate care

                Respond with JSON:
                {{
                    "safe": true/false,
                    "warning": "description if unsafe",
                    "risk_level": "low|medium|high|critical"
                }}
                """
        try:
            response = self.model.generate_response(prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response[json_start:json_end])
        except:
            pass
        return {"safe": True, "warning": "", "risk_level": "low"}

#     def get_workflow_summary(self) -> str:
#         return f"""
# 🔄 **Medical AI Workflow Summary**

# **Patient Context:**
# {self.context.get_context_summary()}

# **Agents Consulted:**
# {', '.join(self.context.specialist_consultations.keys()) if self.context.specialist_consultations else 'Primary care team'}

# **Workflow Steps Completed:**
# {len(self.context.conversation_history)} diagnostic steps

# **Confidence Level:** Based on symptom clarity and agent consensus

# **Next Recommended Actions:**
# - Monitor symptom progression
# - Follow treatment plan as prescribed  
# - Seek immediate care if symptoms worsen
# """
