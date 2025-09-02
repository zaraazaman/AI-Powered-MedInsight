from backend.med_model.model_loader import load_model
from backend.med_model.model_loader import query_medical_qa


def is_medical_question(question):

    medical_keywords = [
        "disease", "disorder", "infection", "bacteria", "virus", "treatment",
        "symptoms", "cancer", "arthritis", "diabetes", "antibiotic", "fever"
    ]
    return any(word in question.lower() for word in medical_keywords)

def answer_medical_question(question: str) -> str:
    if not question.strip():
        return ("⚠️ Please mention your query clearly.")
    return query_medical_qa(question)