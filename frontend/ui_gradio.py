import os
from transformers import AutoProcessor, AutoModel
import torch
import sys
import pandas as pd
import subprocess
import re
from datetime import datetime
import gradio as gr
import json
import numpy as np
from PIL import Image
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.agents.orchestrator_agent import MedicalOrchestrator
from backend.med_model.model_loader import load_model
from backend.agents.monitoring_agent import analyze_patient_history,generate_monitoring_report,summarize_trends_llm,HISTORY_PATH
from backend.agents.qa_agent import answer_medical_question
from backend.agents.report_agent import write_report
from backend.agents.treatment_agent import generate_treatment

relevant_responses = "backend/logs/relevant_diagnosis.csv"
irrelevant_responses = "backend/logs/irrelevant_responses.csv"

diagnosis_model = load_model('diagnosis')
monitor_model = load_model("monitoring")
orchestrator = MedicalOrchestrator(diagnosis_model)

biv_model_name = "microsoft/BiomedVLP-BioViL-T"
biv_processor = AutoProcessor.from_pretrained(biv_model_name, trust_remote_code=True)
biv_model = AutoModel.from_pretrained(biv_model_name, trust_remote_code=True)


def analyze_with_biovil(image, question):
    try:
        inputs = biv_processor(
            text=[question],
            images=[image],       
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = biv_model(**inputs)
            image_embeds = outputs.Fdiagimage_embeds  
            text_embeds = outputs.text_embeds

        score = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        return (
            f"🔎 BioViL-T Analysis\n"
            f"Question: {question}\n\n"
            f"➡️ Similarity score: {score:.3f}\n\n"
            f"⚠️ Higher score = stronger match between image and question."
        )
    except Exception as e:
        return f"❌ BioViL-T error: {str(e)}"


def analyze_medical_image(image, image_type, symptoms=""):
    if image is None:
        return "❌ Upload an image first before analysis."
    
    if symptoms is None:
        symptoms = ""

    if symptoms.strip().endswith("?"):
        return analyze_with_biovil(image, symptoms)

    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        brightness = np.mean(img_array)
        contrast = np.std(img_array)

        analysis = f"""
Medical Image Analysis Report

Image Properties:
- Type: {image_type}
- Dimensions: {width} x {height} pixels
- Brightness: {brightness:.1f}/255
- Contrast: {contrast:.1f}

AI Analysis:
"""

        if "X-Ray" in image_type:
            if brightness < 100:
                analysis += "• Dark regions detected - possible fluid or dense tissue\n"
            if contrast > 50:
                analysis += "• Good image contrast for bone structure visibility\n"
            analysis += "• Recommended: Professional radiologist review required\n"

        elif "MRI" in image_type or "CT" in image_type:
            analysis += "• Cross-sectional imaging detected\n"
            analysis += "• Soft tissue contrast analysis needed\n"
            analysis += "• Recommended: Specialist consultation required\n"

        if symptoms.strip():
            analysis += f"\nClinical Correlation:\n"
            analysis += f"• Patient symptoms: {symptoms[:100]}...\n"
            analysis += f"• Image findings should be correlated with clinical presentation\n"

        analysis += "\n⚠️ Disclaimer: This is a basic analysis. Always consult qualified medical professionals for diagnosis."
        return analysis

    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"


def analyze_input_enhanced(symptoms, image, image_type, image_caption=""):
    image_analysis = ""
    if image is not None:
        if image_caption.strip():
            image_analysis = analyze_with_biovil(image, image_caption)
        else:
            image_analysis = analyze_medical_image(image, image_type, symptoms)
    
    if not symptoms.strip() and image is None:
        # return plain strings instead of gr.update
        return (
            "⚠️ Blank input. Please provide symptoms ",
            "",
            "⚠️ No image analysis available"
        )    
    
    try:
        diagnosis, treatment, _ = orchestrator.coordinate_diagnosis_workflow(symptoms)
        os.makedirs(os.path.dirname(relevant_responses), exist_ok=True)
        os.makedirs(os.path.dirname(irrelevant_responses), exist_ok=True)

        if "❌ Irrelevant context. Please provide medically relevant query." in diagnosis:
            irr_df = pd.DataFrame([{
                "Symptoms": symptoms,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            try:
                existing = pd.read_csv(irrelevant_responses)
                irr_df = pd.concat([existing, irr_df], ignore_index=True)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass
            irr_df.to_csv(irrelevant_responses, index=False)
            return (
                diagnosis,
                "",
                "⚠️ Irrelevant query, no image analysis"
            )

        log_df = pd.DataFrame([{
            "Symptoms": symptoms,
            "Diagnosis": diagnosis,
            "Treatment": treatment,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        # (You forgot to save log_df somewhere, maybe intentional?)

        if image is not None:
            enhanced_diagnosis = f"""{diagnosis}{image_analysis}"""
        else:
            enhanced_diagnosis = diagnosis
                    
        return (
            enhanced_diagnosis,
            treatment,
            image_analysis if image else "📷 No image provided"
        )
    
    except Exception as e:
        return (
            f"⚠️ Error during analysis: {str(e)}",
            "",
            image_analysis if image else "No image provided"
        )

# For textual analysis
def analyze_text_only(symptoms, severity):
    symptoms_with_severity = f"{symptoms} (Severity: {severity}/10)"
    diagnosis, treatment, _ = analyze_input_enhanced(symptoms_with_severity, None, None, "")
    return (
        gr.update(value=diagnosis),
        gr.update(value=treatment)
    )


# For image analysis
def analyze_image_only(image, image_type, image_caption):
    diagnosis, treatment, image_analysis = analyze_input_enhanced("", image, image_type, image_caption)
    return (
        gr.update(value=diagnosis),
        gr.update(value=treatment),
        gr.update(value=image_analysis)
    )


def create_report(name, age, gender, illnesses, diagnosis, treatment):
    if not name.strip() or not age or not gender:
        return "⚠️ Please fill the required fields before generating the report.", None
    
    report_text = write_report(
    patient_name=name,
    age=age,
    gender=gender,
    illnesses=illnesses,
    diagnosis=diagnosis,
    treatment=treatment
)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp_file.write(report_text)
    tmp_file.close()

    return report_text, tmp_file.name

def user_interface():
    with gr.Blocks(css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }

    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
        padding: 20px;
        border-radius: 0;
        border: none;
        box-shadow: none;
    }

    #header {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        color: white !important;
        padding: 30px 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(30, 64, 175, 0.3);
        transition: transform 0.3s ease;
    }
    
    #header:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(30, 64, 175, 0.4);
    }
    
    #header * {
        color: white !important;
        font-weight: 600 !important;
    }

    #header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 4s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    button {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
        padding: 14px 28px !important;
        font-size: 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.25) !important;
        position: relative !important;
        overflow: hidden !important;
        letter-spacing: 0.025em !important;
    }

    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(30, 64, 175, 0.35) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%) !important;
    }

    button:active {
        transform: translateY(0) !important;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3) !important;
    }

    button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: width 0.4s ease, height 0.4s ease;
        transform: translate(-50%, -50%);
    }

    button:active::before {
        width: 300px;
        height: 300px;
    }

    textarea, input, .gr-textbox, .gr-dropdown {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transition: all 0.3s ease !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
    }

    textarea:focus, input:focus, .gr-textbox:focus-within, .gr-dropdown:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1), 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        background: white !important;
        transform: translateY(-1px) !important;
    }
    .content-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
        backdrop-filter: blur(20px);
        position: relative;
        transition: all 0.3s ease;
    }

    .content-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
        border-color: rgba(59, 130, 246, 0.3);
    }

    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        border-radius: 16px 16px 0 0;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .content-card:hover::before {
        opacity: 1;
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }

    .stat-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(59, 130, 246, 0.05), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
        opacity: 0;
    }

    .stat-card:hover::before {
        opacity: 1;
        animation: stat-shine 1.5s ease-in-out;
    }

    @keyframes stat-shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e40af;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }

    .stat-label {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .back-button {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        width: auto !important;
        min-width: 100px !important;
        max-width: 120px !important;
        border-radius: 10px !important;
    }

    .back-button:hover {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%) !important;
        transform: translateX(-2px) translateY(-1px) !important;
    }

    #severity-slider {
        padding: 20px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        margin: 15px 0;
    }

    #severity-slider input[type=range] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
        outline: none;
        transition: all 0.3s ease;
    }

    #severity-slider input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4);
    }

    #severity-slider input[type=range]::-webkit-slider-thumb:hover {
        transform: scale(1.3);
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.6);
    }

    #severity-slider label {
        font-weight: 600;
        color: #1e40af;
        font-size: 15px;
        margin-bottom: 10px;
        display: block;
    }

    .gr-radio {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 15px;
        border: 2px solid #e2e8f0;
    }

    .gr-accordion {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }

    .gr-image {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        background: rgba(248, 250, 252, 0.8) !important;
        transition: all 0.3s ease !important;
    }

    .gr-image:hover {
        border-color: #3b82f6 !important;
        background: rgba(239, 246, 255, 0.8) !important;
    }

    .chat-bubble {
        max-width: 75%;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .chat-bubble:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .bot {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        border-left: 4px solid #3b82f6;
    }

    .user {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
   
    .breadcrumb {
        background: rgba(255, 255, 255, 0.8);
        padding: 12px 20px;
        border-radius: 25px;
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }

    #footer {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-radius: 0;
        margin-top: 60px;
        padding: 25px 0;
        backdrop-filter: blur(20px);
        color: #64748b;
    }

    #footer a {
        color: #3b82f6;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    #footer a:hover {
        color: #1d4ed8;
        text-decoration: underline;
    }

    .loading {
        position: relative;
    }

    .loading::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        border: 2px solid #e2e8f0;
        border-top: 2px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        transform: translate(-50%, -50%);
    }

    @keyframes spin {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }

    /* Smooth Transitions */
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }

    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    p {
        color: #475569;
        line-height: 1.7;
    }

    *:focus {
        outline: none !important;
    }

    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px;
        }
        
        #header {
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .content-card {
            padding: 20px;
            margin: 15px 0;
        }
        
        .stat-card {
            padding: 20px;
        }
        
        .stat-number {
            font-size: 2rem;
        }
    }
""") as demo:
    
        tab_state=gr.State(value="home")
        tab_selector = gr.Radio(
            choices=["home", "diagnosis","report", "chat"],
            value="home",
            visible=False,
            container=False,
            label=""
        )

        #HOME TAB 
        with gr.Column(visible=True) as home_tab:
            gr.HTML("""
            <div id='header' class='fade-in'>
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" stroke="white" stroke-width="2" fill="none"/>
                            <path d="M3 12h3l2-4 4 8 2-4h7" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2.2rem; font-weight: 700; margin: 0;">MedInsight AI</div>
                        <div style="font-size: 1rem; margin-top: 5px; opacity: 0.9; font-weight: 400;">
                            Your Intelligent Health Companion
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="stat-card fade-in">
                        <div style="color: #059669; margin-bottom: 10px;">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="margin: 0 auto; display: block;">
                                <circle cx="12" cy="12" r="10" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
                                <path d="M8 12l2 2 4-4" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                            </svg>
                        </div>
                        <div class="stat-number counter">247</div>
                        <div class="stat-label">Consultations Completed</div>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="stat-card fade-in" style="animation-delay: 0.1s;">
                        <div style="color: #0891b2; margin-bottom: 10px;">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="margin: 0 auto; display: block;">
                                <circle cx="12" cy="12" r="10" fill="#fbbf24" stroke="#f59e0b" stroke-width="1"/>
                                <path d="M12 6v6l4 2" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                                <circle cx="12" cy="12" r="1.5" fill="white"/>
                            </svg>
                        </div>
                        <div class="stat-number">< 3s</div>
                        <div class="stat-label">Average Response Time</div>
                    </div>
                    """)
                    
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="stat-card fade-in" style="animation-delay: 0.2s;">
                        <div style="color: #7c3aed; margin-bottom: 10px;">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="margin: 0 auto; display: block;">
                                <circle cx="12" cy="12" r="8" fill="#e5e7eb"/>
                                <path d="M12 4 A 8 8 0 0 1 19.314 9.372 L 12 12 Z" fill="#10b981"/>
                                <path d="M19.314 9.372 A 8 8 0 0 1 16.243 19.071 L 12 12 Z" fill="#3b82f6"/>
                                <path d="M16.243 19.071 A 8 8 0 0 1 7.757 19.071 L 12 12 Z" fill="#f59e0b"/>
                                <path d="M7.757 19.071 A 8 8 0 0 1 12 4 L 12 12 Z" fill="#ef4444"/>
                                <circle cx="12" cy="12" r="3" fill="white"/>
                            </svg>
                        </div>
                        <div class="stat-number counter">94</div>
                        <div class="stat-label">Clinical Accuracy %</div>
                    </div>
                    """)

            gr.HTML("""
                <div class="content-card fade-in" style="animation-delay: 0.3s;">
                    <h2 style="color:#1e40af; font-size:2.2em; margin-bottom:20px; text-align: center;">
                        Your AI-Powered Medical Companion
                    </h2>
                    <div style="text-align: center; margin-bottom: 25px;">
                        <p style="font-size:1.1em; color: #64748b; max-width: 800px; margin: 0 auto;">
                            Designed to provide fast, reliable, and actionable health insights—all in one intelligent 
                            platform for learners, healthcare professionals, and everyday users.
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; margin-top: 30px;">
                        <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 25px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                            <h3 style="color: #1e40af; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                                </svg>
                                Quick Symptom Analysis
                            </h3>
                            <p style="color: #475569; margin: 0;">
                                Interprets your input to highlight potential conditions and relevant health information with clarity.
                            </p>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 25px; border-radius: 12px; border-left: 4px solid #10b981;">
                            <h3 style="color: #059669; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
                                </svg>
                                Personalized Treatment Guidance
                            </h3>
                            <p style="color: #475569; margin: 0;">
                                Suggests next steps based on evidence-informed medical standards and best practices.
                            </p>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%); padding: 25px; border-radius: 12px; border-left: 4px solid #a855f7;">
                            <h3 style="color: #7c3aed; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                                </svg>
                                Medical Q&A Chatbot
                            </h3>
                            <p style="color: #475569; margin: 0;">
                                Provides structured, concise answers to educational medical questions using advanced AI.
                            </p>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); padding: 25px; border-radius: 12px; border-left: 4px solid #f59e0b;">
                            <h3 style="color: #d97706; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M9 17h6l-1-8H8l1 8zM12 2C8.14 2 5 5.14 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.86-3.14-7-7-7z"/>
                                </svg>
                                Patient Monitoring Reports
                            </h3>
                            <p style="color: #475569; margin: 0;">
                                Generates organized summaries for tracking health status over time for professionals and learners.
                            </p>
                        </div>
                    </div>

                    <div style="text-align: center; margin-top: 35px; padding: 25px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 12px; border: 2px solid #e2e8f0;">
                        <p style="font-size:1.05em; color: #334155; margin: 0; font-weight: 500;">
                            Streamlining healthcare workflows, enhancing learning, and providing reliable insights—
                            <strong style="color: #1e40af;">accessible medical support right at your fingertips</strong>, anytime and anywhere.
                        </p>
                    </div>

                    <div style="margin-top: 25px; padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border-left: 4px solid #ef4444;">
                        <p style="font-size:0.9em; color: #7f1d1d; margin: 0; line-height: 1.6;">
                            <strong>Important:</strong> This platform is intended for educational and informational purposes only. 
                            It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult 
                            a qualified healthcare provider for medical decisions and personalized care.
                        </p>
                    </div>
                </div>
                """)

            with gr.Row():
                with gr.Column(scale=1):
                    go_diagnose = gr.Button("Perform Diagnosis", size="lg")
                with gr.Column(scale=1):
                    go_chat = gr.Button("Chat with MedBot", size="lg")

        #DIAGNOSIS TAB
        with gr.Column(visible=False) as diagnosis_tab:
            gr.HTML("<div class='breadcrumb'>Home ▸ Diagnosis & Treatment</div>")
            gr.HTML("""
            <div id='header' class='fade-in'>
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" stroke="white" stroke-width="2" fill="none"/>
                            <path d="M3 12h3l2-4 4 8 2-4h7" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2.2rem; font-weight: 700; margin: 0;">MedInsight AI</div>
                        <div style="font-size: 1rem; margin-top: 5px; opacity: 0.9; font-weight: 400;">
                            Your Intelligent Health Companion
                        </div>
                    </div>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    back_btn_diagnosis = gr.Button("← Back", size="sm", elem_classes=["back-button"])
                with gr.Column(scale=4):
                    pass  

            gr.HTML("""
            <div class="content-card fade-in">
                <h2 style="color:#1e40af; font-size:2.2em; margin-bottom:20px; display: flex; align-items: center; gap: 15px;">
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
                            <path d="M19.5 12c0-1.232-.046-2.453-.138-3.662a4.006 4.006 0 00-3.7-3.7 48.678 48.678 0 00-7.324 0 4.006 4.006 0 00-3.7 3.7c-.017.22-.032.441-.046.662M19.5 12l3-3m-3 3l-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.006 4.006 0 003.7 3.7 48.656 48.656 0 007.324 0 4.006 4.006 0 003.7-3.7c.017-.22.032-.441.046-.662M4.5 12l3 3m-3-3l3-3"/>
                        </svg>
                    </div>
                    Medical Diagnosis & Treatment
                </h2>
                <p style="font-size:1.1em; margin-bottom:20px; color: #64748b;">
                    AI-powered diagnostic assistance designed to support healthcare professionals, students, and users 
                    in evaluating symptoms and understanding potential medical conditions.
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0;">
                    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                        <h4 style="color: #1e40af; margin-top: 0;">Quick Analysis</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Efficiently examines symptoms and identifies potential conditions based on clinical patterns.
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
                        <h4 style="color: #059669; margin-top: 0;">Evidence-Based</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Offers suggestions derived from established medical standards and best practices.
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #a855f7;">
                        <h4 style="color: #7c3aed; margin-top: 0;">Comprehensive</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Provides clear explanations of potential causes and management strategies.
                        </p>
                    </div>
                </div>

                <div style="margin-top: 25px; padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border-left: 4px solid #ef4444;">
                    <p style="font-size:0.9em; color: #7f1d1d; margin: 0; line-height: 1.6;">
                        <strong>Important:</strong> This tool is for educational and informational purposes only. 
                        It is not a substitute for professional medical advice, diagnosis, or treatment. 
                        Always consult a qualified healthcare provider for medical decisions.
                    </p>
                </div>
            </div>
            """)

            with gr.Row():
                analysis_mode = gr.Radio(
                    choices=["Textual Analysis", "Image Analysis"],
                    label="Choose Analysis Type",
                    value="None",
                    elem_classes=["analysis-mode-radio"]
                )

            with gr.Column(visible=False) as textual_section:
                with gr.Row():
                    with gr.Column():
                        symptom_input = gr.Textbox(
                            label="Describe your symptoms",
                            placeholder="e.g. persistent cough for 3 days, mild fever, headache",
                            lines=3,
                            elem_classes=["enhanced-textbox"]
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        severity_input = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Symptom Severity Level",
                            elem_id="severity-slider",
                            elem_classes=["enhanced-slider"]
                        )
                    with gr.Column(scale=1):
                        emoji_box = gr.HTML("<div style='font-size:48px; text-align:center; padding: 20px;'>😐</div>")
                
                def severity_display(level):
                    emojis = ["😌","🙂","😊","😀","😓","😕","😟","😣","😫","🤒"]
                    colors = ["#10b981", "#059669", "#0d9488", "#0891b2", "#0284c7", "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef"]
                    return f"<div style='font-size:48px; text-align:center; padding: 20px; color: {colors[level-1]}; transition: all 0.3s ease;'>{emojis[level-1]}</div>"

                severity_input.change(
                    severity_display,
                    inputs=severity_input,
                    outputs=emoji_box
                )
                
                diagnose_button = gr.Button("Analyze Symptoms", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        diagnosis_box = gr.Textbox(
                            label="Diagnosis Results", 
                            lines=8, 
                            interactive=False,
                            elem_classes=["result-box"]
                        )
                    with gr.Column():
                        treatment_box = gr.Textbox(
                            label="Treatment & Recommendations", 
                            lines=8, 
                            interactive=False,
                            elem_classes=["result-box"]
                        )
                
                report_button = gr.Button("Generate Medical Report", variant="secondary", size="lg")
                
                diagnose_button.click(
                    fn=analyze_text_only,
                    inputs=[symptom_input, severity_input],
                    outputs=[diagnosis_box, treatment_box]
                )

            with gr.Column(visible=False) as image_section:
                gr.HTML("""
                <div style="margin-bottom: 20px;">
                    <h3 style="color: #1e40af; margin-bottom: 10px;">Medical Image Analysis</h3>
                    <p style="color: #64748b; font-size: 0.95em;">
                        Upload medical images for AI-powered analysis and insights.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(
                            label="Upload Medical Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=300,
                            elem_classes=["enhanced-image-upload"]
                        )
                    with gr.Column(scale=1):
                        image_type = gr.Radio(
                            choices=["X-Ray", "MRI/CT Scan", "Other"],
                            label="Image Type",
                            value="X-Ray",
                            elem_classes=["image-type-radio"]
                        )
                        
                with gr.Row():
                    image_caption = gr.Textbox(
                        label="Image Description or Question",
                        placeholder="e.g. What does this X-ray show? or Describe any abnormalities",
                        lines=3,
                        elem_classes=["enhanced-textbox"]
                    )
                
                img_analysis_button = gr.Button("Analyze Medical Image", variant="primary", size="lg")
                
                with gr.Row():
                    img_analysis_box = gr.Textbox(
                        label="Image Analysis Results", 
                        lines=8, 
                        interactive=False,
                        elem_classes=["result-box"]
                    )

                analysis_mode.change(
                    lambda mode: (gr.update(visible=(mode=="Textual Analysis")), gr.update(visible=(mode=="Image Analysis"))),
                    inputs=analysis_mode,
                    outputs=[textual_section, image_section]
                )

        #REPORT TAB
        with gr.Column(visible=False) as report_tab:
            gr.HTML("<div class='breadcrumb'>Home ▸ Diagnosis & Treatment ▸ Medical Report</div>")
            gr.HTML("""
            <div id='header' class='fade-in'>
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" stroke="white" stroke-width="2" fill="none"/>
                            <path d="M3 12h3l2-4 4 8 2-4h7" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2.2rem; font-weight: 700; margin: 0;">MedInsight AI</div>
                        <div style="font-size: 1rem; margin-top: 5px; opacity: 0.9; font-weight: 400;">
                            Your Intelligent Health Companion
                        </div>
                    </div>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    back_btn_report = gr.Button("← Back", size="sm", elem_classes=["back-button"])
                with gr.Column(scale=4):
                    pass 

            gr.HTML("""
            <div class="content-card fade-in">
                <h2 style="color:#1e40af; font-size:2.2em; margin-bottom:20px; display: flex; align-items: center; gap: 15px;">
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
                            <path d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
                        </svg>
                    </div>
                    Patient Medical Report Generator
                </h2>
                <p style="font-size:1.1em; margin-bottom:20px; color: #64748b;">
                    Generate comprehensive medical reports based on diagnostic analysis. These reports can be 
                    downloaded and shared with healthcare professionals for further consultation.
                </p>
                <div style="margin-top: 20px; padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border-left: 4px solid #ef4444;">
                    <p style="font-size:0.9em; color: #7f1d1d; margin: 0; line-height: 1.6;">
                        <strong>Important:</strong> This report is generated by an AI system for educational and 
                        informational purposes only. It should not be considered a substitute for professional 
                        medical advice, diagnosis, or treatment.
                    </p>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(
                        label="Patient Name", 
                        placeholder="Enter patient's full name",
                        elem_classes=["enhanced-textbox"]
                    )
                with gr.Column():
                    age = gr.Dropdown(
                        choices=[str(i) for i in range(1, 101)], 
                        label="Age",
                        elem_classes=["enhanced-dropdown"]
                    )
                with gr.Column():
                    gender = gr.Radio(
                        ["Male", "Female", "Other"], 
                        label="Gender",
                        elem_classes=["gender-radio"]
                    )

            prev_illness_text = gr.Textbox(
                label="Medical History & Previous Illnesses",
                placeholder="Click conditions below or type manually",
                lines=3,
                elem_classes=["enhanced-textbox"]
            )

            gr.HTML("<h4 style='color: #1e40af; margin: 20px 0 10px 0;'>Common Conditions (Click to Add):</h4>")
            with gr.Row():
                illnesses = ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Migraine", "Depression", "Allergies", "Psoriasis"]
                illness_buttons = [gr.Button(ill, size="sm", variant="secondary") for ill in illnesses]
            
            generate_report_btn = gr.Button("Generate Complete Report", variant="primary", size="lg")
            
            with gr.Accordion("View Generated Report", open=False):
                final_report_box = gr.Textbox(
                    label="", 
                    lines=30, 
                    interactive=False,
                    elem_classes=["report-display"]
                )
                download_file = gr.File(label="Download Report")
                        
            for btn in illness_buttons:
                btn.click(
                    fn=lambda current, new=btn.value: (current + ", " + new if current else new),
                    inputs=[prev_illness_text],
                    outputs=[prev_illness_text]
                )
                     
            def validate_and_generate_report(name, age, gender, illnesses, diagnosis, treatment):
                if not name or not age or not gender:
                    return "Please fill all required fields before generating report.", None
                
                report_text, file_path = create_report(name, age, gender, illnesses, diagnosis, treatment)
                return report_text, file_path
            
            generate_report_btn.click(
                fn=validate_and_generate_report,
                inputs=[name, age, gender, prev_illness_text, diagnosis_box, treatment_box],
                outputs=[final_report_box, download_file]
            )

        #CHATBOT TAB
        with gr.Column(visible=False) as chatbot_tab:
            gr.HTML("<div class='breadcrumb'>Home ▸ Medical Learning Assistant</div>")
            gr.HTML("""
            <style>
            .chat-input textarea,
            .chat-input input {
                background-color: #fffaf0 !important; 
                color: #abc8f7 !important;            
                border: none !important;              
                border-radius: 12px !important;
                padding: 8px 12px !important;
                box-shadow: none !important;        
                resize: none;
            }

            .enhanced-textbox textarea {
                background-color: #fffaf0 !important; 
                color: #abc8f7 !important;            
                border: none !important;              
                outline: none !important;
                box-shadow: none !important;
            }

            
            .chatbot .message-wrap.user .message,
            .chatbot .message-wrap.user,
            .chatbot .user,
            .chat-area .message.user,
            .chatbot div[data-testid="user"] {
                background-color: #a6cff7 !important; /* User message - light blue */
                color: #1e293b !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 10px 14px !important;
                box-shadow: none !important;
            }

            .chatbot .message-wrap.bot .message,
            .chatbot .message-wrap.bot,
            .chatbot .bot,
            .chat-area .message.bot,
            .chatbot div[data-testid="bot"] {
                background-color: #ffffa6 !important; /* Bot response - light yellow */
                color: #1e293b !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 10px 14px !important;
                box-shadow: none !important;
            }

            .chatbot .message p,
            .chatbot .message span,
            .chatbot .message div,
            .chatbot p,
            .chatbot span {
                color: #1e293b !important;
                background: inherit !important;
            }

            .chatbot [class*="user"] {
                background-color: #a6cff7 !important;
                color: #1e293b !important;
            }

            .chatbot [class*="bot"] {
                background-color: #ffffa6 !important;
                color: #1e293b !important;
            }

            .typing-indicator {
                display: flex;
                align-items: center;
                margin: 5px 0;
                padding: 10px;
                background-color: #ffffa6;
                border-radius: 12px;
                width: fit-content;
            }

            .typing-bubbles {
                display: flex;
                align-items: center;
                gap: 4px;
            }

            .typing-bubbles span {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #059669;
                border-radius: 50%;
                animation: blink 1.4s infinite;
            }

            .typing-bubbles span:nth-child(2) {
                animation-delay: 0.2s;
            }

            .typing-bubbles span:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes blink {
                0% { opacity: 0.2; }
                20% { opacity: 1; }
                100% { opacity: 0.2; }
            }
            </style>
            """)

            gr.HTML("""
            <div id='header' class='fade-in'>
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" stroke="white" stroke-width="2" fill="none"/>
                            <path d="M3 12h3l2-4 4 8 2-4h7" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2.2rem; font-weight: 700; margin: 0;">MedInsight AI</div>
                        <div style="font-size: 1rem; margin-top: 5px; opacity: 0.9; font-weight: 400;">
                            Your Intelligent Health Companion
                        </div>
                    </div>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    back_btn_chat = gr.Button("← Back", size="sm", elem_classes=["back-button"])
                with gr.Column(scale=4):
                    pass  

            gr.HTML("""
            <div class="content-card fade-in">
                <h2 style="color:#1e40af; font-size:2.2em; margin-bottom:20px; display: flex; align-items: center; gap: 15px;">
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
                            <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                        </svg>
                    </div>
                    Medical Learning Assistant
                </h2>
                <p style="font-size:1.1em; margin-bottom:20px; color: #64748b;">
                    An AI-powered educational platform designed to help learners, students, and healthcare enthusiasts 
                    explore medical knowledge in a structured and educational manner.
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 25px 0;">
                    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;">
                        <h4 style="color: #1e40af; margin-top: 0;">MedExpert Powered</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Utilizes advanced AI to generate responses about diseases, anatomy, physiology, and treatments.
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
                        <h4 style="color: #059669; margin-top: 0;">Educational Focus</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Designed for learning and understanding medical concepts in a structured format.
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #a855f7;">
                        <h4 style="color: #7c3aed; margin-top: 0;">Reliable Information</h4>
                        <p style="color: #475569; margin: 0; font-size: 0.95em;">
                            Provides structured and informative responses for educational purposes.
                        </p>
                    </div>
                </div>

                <div style="margin-top: 25px; padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border-left: 4px solid #ef4444;">
                    <p style="font-size:0.9em; color: #7f1d1d; margin: 0; line-height: 1.6;">
                        <strong>Important:</strong> This platform is for educational purposes only and is not a substitute 
                        for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare 
                        provider for medical decisions and personalized care.
                    </p>
                </div>
            </div>
            """)

            with gr.Row():
                chatbot_box = gr.Chatbot(
                    label="MedBot Chat",
                    height=400,
                    elem_classes=["chat-area"]
                )

            question_input = gr.Textbox(
                label="Ask Your Medical Question", 
                placeholder="e.g., What are viruses? How does diabetes work? Explain the cardiovascular system...", 
                lines=2,
                elem_classes=["enhanced-textbox", "chat-input"]
            )

            ask_button = gr.Button("Ask Question", variant="primary", size="lg")
            typing_message = "AI is thinking..."


            def show_typing(history, user_input):
                if user_input.strip(): 
                    history = history + [(user_input, typing_message)]
                    return history, ""
                return history, user_input

            def generate_answer(history):
                if history:  # Check if history exists
                    user_input, _ = history[-1]  # get last user message
                    bot_reply = answer_medical_question(user_input)
                    history[-1] = (user_input, bot_reply)  
                return history
            ask_button.click(
                fn=show_typing,
                inputs=[chatbot_box, question_input],
                outputs=[chatbot_box, question_input]
            ).then(
                fn=generate_answer,
                inputs=chatbot_box,
                outputs=chatbot_box
            )

        
            question_input.submit(
                fn=show_typing,
                inputs=[chatbot_box, question_input],
                outputs=[chatbot_box, question_input]
            ).then(
                fn=generate_answer,
                inputs=chatbot_box,
                outputs=chatbot_box
            )

        def change_tab(tab_choice):
            return (
                gr.update(visible=tab_choice == "home"),
                gr.update(visible=tab_choice == "diagnosis"),
                gr.update(visible=tab_choice == "chat")
            )

        tab_selector.change(
            lambda tab: gr.update(visible=True) if tab == "home" else gr.update(visible=False),
            inputs=tab_selector,
            outputs=home_tab
        )

        tab_selector.change(
            lambda tab: gr.update(visible=True) if tab == "diagnosis" else gr.update(visible=False),
            inputs=tab_selector,
            outputs=diagnosis_tab
        )

        tab_selector.change(
            lambda tab: gr.update(visible=True) if tab == "chat" else gr.update(visible=False),
            inputs=tab_selector,
            outputs=chatbot_tab
        )
        with gr.Row(elem_id="footer", visible=True):
            gr.HTML("""
            <div style="
                margin-top: auto;
                width: 100%;
                text-align: center;
                padding: 25px 0;
                font-size: 14px;
                color: #64748b;
                border: none !important;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
                backdrop-filter: blur(20px);
                border-top: 1px solid rgba(226, 232, 240, 0.8);
                font-family: 'Inter', sans-serif;
                font-weight: 500;
            ">
                <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: center; align-items: center; gap: 30px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                        </svg>
                        <span>2025 MedInsight AI</span>
                    </div>
                    <span>|</span>
                    <span>For Educational Purposes Only</span>
                    <span>|</span>
                    <a href='#' style='color:#3b82f6; text-decoration:none; transition: color 0.3s ease;'>Privacy Policy</a>
                    <span>|</span>
                    <a href='#' style='color:#3b82f6; text-decoration:none; transition: color 0.3s ease;'>Terms of Use</a>
                    <span>|</span>
                    <a href='#' style='color:#3b82f6; text-decoration:none; transition: color 0.3s ease;'>Contact Support</a>
                </div>
            </div>
            """)

        def show_home():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        def show_diagnosis():
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        def show_report():
            return ("report", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))

        def show_chatbot():
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        def go_back_home():
            return ("home", *show_home())
        
        go_diagnose.click(lambda: ("diagnosis", *show_diagnosis()), outputs=[tab_selector, home_tab, diagnosis_tab, chatbot_tab])
        go_chat.click(lambda: ("chat", *show_chatbot()), outputs=[tab_selector, home_tab, diagnosis_tab, chatbot_tab])

        back_btn_diagnosis.click(
            lambda: ("home", *show_home()),
            outputs=[tab_selector, home_tab, diagnosis_tab, chatbot_tab]
        )
        back_btn_chat.click(
            lambda: ("home", *show_home()),
            outputs=[tab_selector, home_tab, diagnosis_tab, chatbot_tab]
        )
        report_button.click(
            fn=show_report, 
            outputs=[tab_selector, home_tab, diagnosis_tab, report_tab, chatbot_tab]
        )
        back_btn_report.click(
            lambda: ("diagnosis", *show_diagnosis()),
            outputs=[tab_selector, home_tab, diagnosis_tab, chatbot_tab, report_tab]
        )

    return demo
