import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from backend.med_model.model_loader import load_model

HISTORY_PATH = "backend/patient_data/history.csv"
REPORTS_DIR = "backend/reports/"
os.makedirs(REPORTS_DIR, exist_ok=True)

def update_health_log(patient_id, vitals_dict):
    """Append new vitals to the patient's health history."""
    df = pd.DataFrame([{
        "patient_id": patient_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **vitals_dict
    }])
    if not os.path.exists(HISTORY_PATH):
        df.to_csv(HISTORY_PATH, index=False)
    else:
        df.to_csv(HISTORY_PATH, mode='a', header=False, index=False)

def analyze_patient_history(patient_id):
    """Return a summary and a chart of the patient's health trends."""
    if not os.path.exists(HISTORY_PATH):
        return "No health data found.", None

    df = pd.read_csv(HISTORY_PATH)
    df = df[df["patient_id"] == patient_id]

    if df.empty:
        return "No entries found for this patient.", None

    summary = summarize_trends_llm(df)
    chart_path = generate_health_chart(df, patient_id)
    return summary, chart_path

def generate_health_chart(df, patient_id):
    """Generate a line plot of patient vitals."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    plt.figure(figsize=(10, 5))
    if 'heart_rate' in df.columns:
        sns.lineplot(x='timestamp', y='heart_rate', data=df, label='Heart Rate')
    if 'temperature' in df.columns:
        sns.lineplot(x='timestamp', y='temperature', data=df, label='Temperature')
    if 'blood_pressure' in df.columns:
        pass  # You can add logic to visualize blood_pressure if it’s structured

    plt.title(f"Vitals Trend: {patient_id}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join(REPORTS_DIR, f"{patient_id}_monitoring_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def summarize_trends_llm(df):
    if not isinstance(df, pd.DataFrame):
        return "⚠️ Invalid data format for monitoring. Expected a DataFrame."

    model = load_model("monitoring")
    try:
        recent = df.tail(3).to_dict(orient="records")
    except Exception as e:
        return f"⚠️ Error extracting recent data: {e}"

    prompt = (
        "You are a medical assistant. Analyze the following health logs and summarize the patient's current condition.\n\n"
        f"{recent}\n\n"
        "Provide any patterns, improvements, or worsening symptoms. "
        "Also provide further guidelines and recommendations to the patient based on their health."
    )

    try:
        response = model.generate_response(prompt)
        return response.strip()
    except Exception as e:
        return f"⚠️ LLM Error: {e}"

def generate_monitoring_report(patient_id: str) -> str:
    if not os.path.exists(HISTORY_PATH):
        return "❌ No health data available."

    df = pd.read_csv(HISTORY_PATH)
    df = df[df["patient_id"] == patient_id]

    if df.empty:
        return "❌ No entries found for this patient."

    return summarize_trends_llm(df)
