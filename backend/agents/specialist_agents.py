class CardiologyAgent:
    """Specialized agent for cardiovascular conditions"""
    
    def __init__(self, model):
        self.model = model
        self.specialty = "Cardiology"
        
    def analyze_cardiovascular_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from cardiovascular perspective"""
        
        patient_context = ""
        if patient_data:
            patient_context = f"""
                     Patient Background:
                     - Age: {patient_data.get('age', 'Unknown')}
                     - Gender: {patient_data.get('gender', 'Unknown')}  
                     - BP History: {patient_data.get('bp_readings', 'No data')}
                     - Current Medications: {patient_data.get('medications', 'None listed')}
                     - Cardiac History: {patient_data.get('cardiac_history', 'No prior cardiac issues')}
                     """
                           
            prompt = f"""
      You are a board-certified cardiologist with 20+ years of experience. Analyze these symptoms for cardiovascular conditions.

      {patient_context}

      Current Symptoms: {symptoms}

      Provide a detailed cardiovascular assessment:

      1. **Cardiac Risk Stratification:**
         - Low/Medium/High cardiovascular risk
         - Risk factors present

      2. **Differential Diagnosis:**
         - Most likely cardiac conditions
         - Rule-out conditions requiring immediate attention

      3. **Recommended Cardiac Workup:**
         - Essential tests (ECG, Echo, Stress test, etc.)
         - Laboratory studies
         - Imaging if indicated

      4. **Immediate Actions:**
         - Any urgent interventions needed
         - When to seek emergency care

      5. **Cardiovascular Management:**
         - Medication considerations
         - Lifestyle modifications
         - Follow-up recommendations

      Focus on actionable, evidence-based recommendations.
      """
        
        return self.model.generate_response(prompt)
    
    def assess_chest_pain(self, symptoms: str, patient_data: dict = None) -> str:
        """Specialized chest pain assessment"""
        
        prompt = f"""
You are evaluating a patient with chest pain. Use the standard chest pain assessment protocol.

Patient Symptoms: {symptoms}
Patient Data: {patient_data if patient_data else 'Limited data available'}

Provide chest pain assessment using HEART Score approach:

1. **History Assessment:**
   - Typical/Atypical/Non-cardiac chest pain
   - Pain characteristics (location, radiation, quality, timing)

2. **Risk Stratification:**
   - Age and risk factors
   - Associated symptoms
   - Hemodynamic stability

3. **Diagnostic Approach:**
   - ECG interpretation needs
   - Cardiac biomarkers
   - Imaging requirements

4. **Disposition:**
   - Emergency department vs outpatient
   - Cardiology consultation urgency
   - Safe discharge criteria

5. **Treatment Recommendations:**
   - Immediate interventions
   - Medications
   - Activity restrictions
"""
        
        return self.model.generate_response(prompt)


class NeurologyAgent:
    """Specialized agent for neurological conditions"""
    
    def __init__(self, model):
        self.model = model
        self.specialty = "Neurology"
        
    def analyze_neurological_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from neurological perspective"""
        
        prompt = f"""
You are a board-certified neurologist. Analyze these symptoms for neurological conditions.

Patient Symptoms: {symptoms}
Patient History: {patient_data if patient_data else 'Limited history available'}

Provide comprehensive neurological assessment:

1. **Neurological Localization:**
   - Central vs peripheral nervous system
   - Anatomical localization if possible
   - Lateralization

2. **Differential Diagnosis:**
   - Most likely neurological conditions
   - Red flag conditions requiring immediate attention
   - Mimic conditions to consider

3. **Neurological Examination Focus:**
   - Key examination components needed
   - Expected findings
   - Concerning signs to monitor

4. **Diagnostic Workup:**
   - Neuroimaging indications (CT, MRI)
   - Laboratory studies
   - Specialized tests (EEG, EMG, LP)

5. **Management Approach:**
   - Acute interventions if needed
   - Neurological medications
   - Referral recommendations
   - Follow-up timeline

Focus on systematic, evidence-based neurological approach.
"""
        
        return self.model.generate_response(prompt)
    
    def assess_headache(self, symptoms: str, patient_data: dict = None) -> str:
        """Specialized headache assessment"""
        
        prompt = f"""
                  You are conducting a headache evaluation using systematic approach.

                  Symptoms: {symptoms}
                  Patient Data: {patient_data if patient_data else 'Limited data'}

                  Provide headache assessment:

                  1. **Headache Classification:**
                     - Primary vs secondary headache
                     - Headache type (tension, migraine, cluster, etc.)
                     - Red flag features

                  2. **SNNOOP10 Assessment:**
                     - Systemic symptoms/signs
                     - Neurologic symptoms/signs  
                     - Onset sudden
                     - Older age (>50)
                     - Pattern change
                     - Positional
                     - Precipitated by Valsalva
                     - Papilledema
                     - Progressive
                     - Pregnancy

                  3. **Diagnostic Approach:**
                     - Imaging indications
                     - Laboratory workup
                     - Specialized testing

                  4. **Treatment Strategy:**
                     - Acute management
                     - Preventive therapy
                     - Lifestyle modifications
                     - When to refer urgently
                  """
                        
        return self.model.generate_response(prompt)


class PharmacologyAgent:
    """Specialized agent for medication management and drug interactions"""
    
    def __init__(self, model):
        self.model = model
        self.specialty = "Clinical Pharmacology"
        
    def analyze_medication_safety(self, medications: list, symptoms: str, patient_data: dict = None) -> str:
        """Analyze medication safety and interactions"""
        
        med_list = ', '.join(medications) if medications else 'No current medications'
        
        prompt = f"""
                        You are a clinical pharmacist conducting medication therapy management.

                        Current Medications: {med_list}
                        Patient Symptoms: {symptoms}
                        Patient Data: {patient_data if patient_data else 'Limited data available'}

                        Provide comprehensive medication analysis:

                        1. **Drug Interaction Assessment:**
                           - Major drug-drug interactions
                           - Drug-disease interactions
                           - Drug-food interactions
                           - Clinical significance of interactions

                        2. **Adverse Drug Reaction Evaluation:**
                           - Possible medication-related symptoms
                           - Timing relationship
                           - Dose-response relationship
                           - Alternative explanations

                        3. **Medication Optimization:**
                           - Dosing appropriateness
                           - Therapeutic alternatives
                           - Cost-effective options
                           - Patient-specific considerations

                        4. **Safety Monitoring:**
                           - Laboratory monitoring needs
                           - Clinical parameters to follow
                           - Signs/symptoms to watch for
                           - Frequency of monitoring

                        5. **Patient Education Points:**
                           - Important medication counseling
                           - Adherence strategies
                           - When to contact healthcare provider
                           - Storage and administration

                        Prioritize patient safety and evidence-based recommendations.
                        """
        
        return self.model.generate_response(prompt)
    
    def recommend_medication_therapy(self, diagnosis: str, patient_data: dict = None) -> str:
        """Recommend evidence-based medication therapy"""
        
        prompt = f"""
                     You are developing an evidence-based medication therapy plan.

                     Diagnosis: {diagnosis}
                     Patient Factors: {patient_data if patient_data else 'Standard adult patient'}

                     Provide medication therapy recommendations:

                     1. **First-Line Therapy:**
                        - Preferred medication(s)
                        - Dosing and administration
                        - Duration of therapy
                        - Evidence level

                     2. **Alternative Options:**
                        - Second-line choices
                        - When to consider alternatives
                        - Contraindications

                     3. **Patient-Specific Considerations:**
                        - Age-related adjustments
                        - Renal/hepatic considerations
                        - Drug allergies/intolerances
                        - Comorbidity considerations

                     4. **Monitoring Plan:**
                        - Efficacy monitoring
                        - Safety monitoring
                        - Laboratory follow-up
                        - Clinical endpoints

                     5. **Patient Counseling:**
                        - How to take medication
                        - Expected benefits
                        - Potential side effects
                        - When to contact provider

                     Base recommendations on current clinical guidelines and evidence.
                     """
        
        return self.model.generate_response(prompt)

class PsychiatryAgent:
    """Specialized agent for mental health and psychiatric evaluation"""

    def __init__(self, model):
        self.model = model
        self.specialty = "Psychiatry"

    def analyze_psychiatric_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from psychiatric perspective"""

        prompt = f"""
                     You are a licensed psychiatrist with experience in diagnosing mental health conditions.

                     Patient Symptoms: {symptoms}
                     Patient History: {patient_data if patient_data else 'Limited history available'}

                     Provide a psychiatric assessment:

                     1. **Primary Concern:**
                        - Likely psychiatric diagnosis or condition

                     2. **Associated Symptoms:**
                        - Emotional, cognitive, behavioral aspects to note

                     3. **Risk Factors:**
                        - Self-harm, suicide risk, trauma history

                     4. **Diagnostic Considerations:**
                        - Screening tools or clinical criteria involved (e.g., DSM-5)

                     5. **Management Recommendations:**
                        - Psychotherapy suggestions
                        - Pharmacological options
                        - Follow-up plan and referrals
                     """
        return self.model.generate_response(prompt)


class PulmonologyAgent:
    """Specialized agent for respiratory and lung-related conditions"""

    def __init__(self, model):
        self.model = model
        self.specialty = "Pulmonology"

    def analyze_respiratory_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from respiratory perspective"""

        prompt = f"""
                  You are a board-certified pulmonologist. Evaluate the following respiratory symptoms.

                  Symptoms: {symptoms}
                  Patient Data: {patient_data if patient_data else 'Limited information'}

                  Provide a comprehensive respiratory assessment:

                  1. **Possible Diagnoses:**
                     - Asthma, COPD, pneumonia, infections, or others
                     - Rule out critical respiratory issues

                  2. **Pulmonary Red Flags:**
                     - Severe shortness of breath, hypoxia, chest pain

                  3. **Recommended Tests:**
                     - Chest X-ray, spirometry, oxygen saturation, CBC, etc.

                  4. **Management Plan:**
                     - Medications (e.g., bronchodilators, steroids, antibiotics)
                     - Oxygen therapy or nebulization if needed
                     - Hospitalization criteria

                  5. **Follow-Up:**
                     - Timeline for reassessment
                     - Preventive advice and lifestyle recommendations
                  """
        return self.model.generate_response(prompt)


class GastroenterologyAgent:
    """Specialized agent for digestive system issues"""

    def __init__(self, model):
        self.model = model
        self.specialty = "Gastroenterology"

    def analyze_digestive_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from gastroenterological perspective"""

        prompt = f"""
                  You are a board-certified gastroenterologist. Analyze the following symptoms related to digestive health.

                  Symptoms: {symptoms}
                  Patient Data: {patient_data if patient_data else 'Limited data available'}

                  Provide a comprehensive GI assessment:

                  1. **Possible Diagnoses:**
                     - Functional or structural GI disorders (e.g., IBS, GERD, ulcers, infections)

                  2. **Red Flags:**
                     - Bleeding, unintended weight loss, severe pain, persistent vomiting

                  3. **Diagnostic Recommendations:**
                     - Endoscopy, colonoscopy, stool tests, abdominal ultrasound/CT

                  4. **Management Plan:**
                     - Medications (antacids, antibiotics, laxatives, etc.)
                     - Dietary modifications
                     - Follow-up needs

                  5. **When to Refer:**
                     - Emergency symptoms
                     - Need for surgical or subspecialty evaluation
                  """
        return self.model.generate_response(prompt)

class DermatologyAgent:
    """Specialized agent for skin-related conditions"""

    def __init__(self, model):
        self.model = model
        self.specialty = "Dermatology"

    def analyze_dermatological_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from a dermatological perspective"""

        prompt = f"""
                  You are a board-certified dermatologist. Analyze the following skin-related symptoms.

                  Symptoms: {symptoms}
                  Patient Data: {patient_data if patient_data else 'Limited data available'}

                  Provide a comprehensive dermatological assessment:

                  1. **Possible Diagnoses:**
                     - Eczema, psoriasis, fungal infections, bacterial skin infections, dermatitis, etc.

                  2. **Concerning Features:**
                     - Rapid progression, signs of infection, systemic symptoms

                  3. **Diagnostic Steps:**
                     - Skin biopsy, culture, allergy testing, Wood's lamp exam

                  4. **Management Recommendations:**
                     - Topical treatments (steroids, antifungals, antibiotics)
                     - Oral medications if needed
                     - Hygiene and skincare routines

                  5. **Referral/Emergency Criteria:**
                     - When to refer to a dermatology clinic
                     - Red flag symptoms needing urgent attention
                  """
        return self.model.generate_response(prompt)

class EndocrinologyAgent:
    """Specialized agent for endocrine system disorders (e.g., diabetes, thyroid issues)"""

    def __init__(self, model):
        self.model = model
        self.specialty = "Endocrinology"

    def analyze_endocrine_symptoms(self, symptoms: str, patient_data: dict = None) -> str:
        """Analyze symptoms from an endocrine perspective"""

        prompt = f"""
You are a board-certified endocrinologist. Analyze the following symptoms from an endocrine perspective.

Symptoms: {symptoms}
Patient Data: {patient_data if patient_data else 'Limited data available'}

Provide a structured assessment:

1. **Possible Endocrine Disorders:**
   - Diabetes mellitus, thyroid dysfunction, adrenal disorders, pituitary abnormalities, etc.

2. **Key Diagnostic Indicators:**
   - Symptoms duration, hormonal imbalance signs, metabolic red flags

3. **Recommended Investigations:**
   - Blood glucose, HbA1c, TSH, T3/T4, cortisol, ACTH, insulin, etc.

4. **Initial Management Plan:**
   - Medications, hormone therapy, lifestyle advice

5. **Referral or Follow-up:**
   - When to refer to an endocrinology clinic
   - Follow-up recommendations based on condition
"""
        return self.model.generate_response(prompt)



class EmergencyAgent:
    """Specialized agent for emergency/urgent conditions"""
    
    def __init__(self, model):
        self.model = model
        self.specialty = "Emergency Medicine"
        
    def triage_urgency(self, symptoms: str) -> dict:
        """Triage patient urgency level"""
        
        prompt = f"""
You are an emergency medicine physician conducting initial triage.

Patient Symptoms: {symptoms}

Assess urgency level and provide triage decision:

1. **Urgency Level:**
   - IMMEDIATE (life-threatening, <15 minutes)
   - URGENT (serious but stable, <1 hour)  
   - LESS URGENT (stable, <4 hours)
   - NON-URGENT (routine care)

2. **Red Flag Assessment:**
   - Life-threatening features present
   - Vital sign concerns
   - Neurological emergencies
   - Cardiovascular emergencies

3. **Disposition:**
   - Emergency department immediately
   - Urgent care center
   - Primary care same day
   - Routine appointment

4. **Immediate Actions:**
   - Call 911 indications
   - First aid measures
   - What NOT to do
   - Information to gather

Respond in JSON format:
{{
    "urgency": "IMMEDIATE|URGENT|LESS_URGENT|NON_URGENT",
    "disposition": "911|ED|URGENT_CARE|PRIMARY_CARE",
    "red_flags": ["flag1", "flag2"],
    "immediate_actions": ["action1", "action2"]
}}
"""
        
        response = self.model.generate_response(prompt)
        
        # Try to parse JSON response
        try:
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                return json.loads(response[json_start:json_end])
        except:
            pass
            
        # Fallback response
        return {
            "urgency": "LESS_URGENT",
            "disposition": "PRIMARY_CARE", 
            "red_flags": [],
            "immediate_actions": ["Monitor symptoms", "Seek appropriate care"]
        } 