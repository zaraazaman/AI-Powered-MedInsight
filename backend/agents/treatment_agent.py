import re
from backend.med_model.model_loader import load_model  
def clean_model_output(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(([^)]*(confidence|citation|reference)[^)]*)\)', '', text, flags=re.IGNORECASE)
    return text.strip()

def generate_treatment(symptoms, diagnosis, model=None):
    model = model or load_model("treatment")
    prompt = f"""
You are a medical treatment recommendation assistant
A patient presents with:
- Symptoms: {symptoms}
- Diagnosis: {diagnosis}

Generate a clear, formatted treatment plan using markdown-like structure. Include the following sections:

1. Primary Medication: List drug names and dosage and frequencies (e.g., "Acetaminophen 325mg every 6 hours").
2. Duration: How long to take each medication and Instructions: Timing, with/without food, etc.
3. Non-Pharmacological Advice: Lifestyle changes, hydration, diet, etc.
4. Recommended Tests (if any): Only if relevant.
5. Follow-Up: What should be done next.

Avoid:
- References or citations
- Confidence scores or disclaimers
- Vague generalizations

"""
    reponse=model.generate_response(prompt)
    return reponse

