import requests
OLLAMA_URL = "http://localhost:11434/api/generate"

class OllamaModel:
    def __init__(self, model_name):
        self.model_name = model_name

    
    def generate_response(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]



def load_model(task_type):
    if task_type in ["diagnosis", "treatment"]:
        return OllamaModel("OussamaELALLAM/MedExpert") 
    elif task_type in ["monitoring", "report"]:
        return OllamaModel("potaTOES33/healthmateai") 
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

def query_medical_qa(question: str) -> str:
    prompt = (
        "You are a medical expert. Answer the following question clearly and concisely:\n\n"
        f"Question: {question}\nAnswer:"
    )

    payload = {
        "model": "OussamaELALLAM/MedExpert",  
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]
