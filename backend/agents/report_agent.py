from datetime import datetime   #to mention date time too with the generated report

def write_report(patient_name: str, age: str, gender: str,
                 illnesses: str, diagnosis: str, treatment: str) -> str:
    if not patient_name or not age or not gender:
        return "⚠️ Cannot generate report. Missing patient details."

    if diagnosis:
        diagnosis_points = "\n".join([f"- {line.strip()}" for line in diagnosis.split('.') if line.strip()])
    else:
        diagnosis_points = "Not provided"
    if treatment:
        treatment_points = "\n".join([f"- {line.strip()}" for line in treatment.split('.') if line.strip()])
    else:
        treatment_points = "Not provided"

    report = f"""
=========================================================================================================================================
                                                                                    MEDICAL REPORT
========================================================================================================================================= 

**Patient Information**
- **Name**   : {patient_name}
- **Age**    : {age}
- **Gender** : {gender}
- **History**: {illnesses if illnesses else "None"}
- **Date**   : {datetime.now().strftime("%Y-%m-%d %H:%M")}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Diagnosis**
{diagnosis_points}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Treatment & Recommendations**
{treatment_points}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Attestion by a Medical Professional:_____________________________
**Notes**
This medical report has been automatically generated using AI.
Please consult a qualified physician for confirmation and further advice.

=========================================================================================================================================
                                                                                    MedInsight AI – Your Health Companion
=========================================================================================================================================
"""
    return report
