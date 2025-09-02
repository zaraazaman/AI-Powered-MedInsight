from backend.med_model.model_loader import load_model


med_model = load_model("diagnosis")

def is_input_medical(input_text, task_type=None): 

    prompt = f"""
You are a medical assistant. Analyze the following input and determine if it is medically relevant or not.

Respond ONLY with:
- "Yes" if it's clearly about a symptom, health issue, diagnosis, treatment, or medical concern
- "No" otherwise.

Input: "{input_text}"
"""
    try:
        response = med_model.generate_response(prompt).strip().lower()
        print("[Verifier] model response:", response)
        return response.startswith("yes")
    except Exception as e:
        print(f"[Verifier] Error during input relevance check: {e}")
        return False


def generate_diagnosis(input_text):
    model = load_model("diagnosis")
    prompt = f"""
You are an expert medical assistant. Read the following patient input and provide a detailed possible diagnosis.

Patient input: "{input_text}"

Return the response in a human-friendly paragraph.
"""
    try:
        response = med_model.generate_response(prompt).strip()
        print("model response:", response)
        return response
    except Exception as e:
        print(f"[Diagnosis] Error during diagnosis generation: {e}")
        return "⚠️ Sorry, there was an error generating the diagnosis."
