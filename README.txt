user input (symptoms + diseases)

            ↓

analyze_input_ehnahced:
    ->checks whether user wrote something
    ->to proceed to diagnosis and treatment generation it goes to the orchestrator agent (coordinate_diagnosis_workflow) ----its inputs (symptoms)

            ↓

fowards the call to diagnosis_agent
    ->loads the LLM MedExpert
    ->checks if the user input is medically relevant (calls the is_input_medical function)---input for this function (model+symptoms)
    ->model responds with yes or no and the answer is returned 

            ↓

the is_input_medical returns back to orchestrator agent(coordinate_diagnosis_workflow)

            ↓

from there called diagnosis_agent(generate diagnosis)---input for this function is the the relevancy results from the prev function + symptoms + model
    ->loads the LLM MedExpert
    ->model generates a corresponding response
    ->returns the generated ans back to orchestrator

            ↓

orchestrator(determine_specialist_consultation) called---input of the function is symptoms + diagnossi generated
    ->this function extracts keywords from the symptoms and each speicalist agent ahs keyowrds defined with it
    ->those keywords are matched with the keyowkrds in the symptoms and a a specialist is forwarded (i.e neurology, radiology)
    ->return back to coordainte_diag_workflow (orchestrator)

            ↓

orchestrator calls the specialist_agent

            ↓

specialist_agent: (calls the calss with the specific medical domain)
    ->that particular special agent performs diagnosis of its own domain
    ->each specail agent is given its own prompt
    ---for example if the input is related to cardiology then cardiologyspecial agent will be consulted, the LLM will generate diagnosis by considering that the symptoms are related to heart etc
    ---the LLM will fetch knowledge related to only cardiology and side by side generate diagnosis and recommendations
    ->the reponse will be returned back to the orchestrator

            ↓

orchestrator (coordainte_diag_workflow)
it will send a fucntion call to the treatment agent

            ↓

treatment_agent function (generate_treatment is triggered)---inputs given to the function are the diagnosis generated + symptoms + model
    ->model is loaded
    ->a treatmentrelated prompt is given to the LLM
    ->it generates a response considering the generated diagnosis
    ->additionallly a treatment_agent function (clean_model_output) represents the generated treatment in a structured and clean manner

            ↓

returns to orchestrator agent
-it validates the generated treatment (checks if the medications or recs arent harmful in any way)
-generates a warning or alert if the situation is critical

            ↓

orchestrator then formats the generated daignosi and treatment and returns back to the frontend to display all the text

            ↓
            
the relevant results are stored in a csv file
the irrelevant results are stored in a separated csv file + their results
