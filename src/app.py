from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form

# Import necessary functions from your model directory
from tensorflow.keras.models import load_model # If using Keras
# You will need to replace these with the actual paths and function names
from src.model.model_training import train_model # Assuming train_model exists
from src.model.model_evaluation import * # Assuming evaluate_model exists
from src.model.feature_engineering import * # Assuming preprocess_sequence exists
# You'll also need to import functions to load your model and potentially process sequences/FASTA
# Example:
# from tensorflow.keras.models import load_model # If using Keras
# from src.data_handling.pdb_processor import process_pdb_file # If processing PDB
# from src.data_handling.ncbi_processor import process_ncbi_sequence # If processing NCBI sequences

# Global variable to hold the loaded model
model = None

app = FastAPI()

# Function to load the model
@app.on_event("startup")
async def load_trained_model():
    global model
    model_path = "src/models/protein_predictor_v1.0.0.h5" # Path relative to project root
    model = load_model(model_path)

app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model_info", response_class=HTMLResponse)
async def model_info(request: Request):
    # In a real application, you would load and display model information here
    return HTMLResponse(content="<h1>Model Information</h1><p>Details about the model structure and parameters.</p>")

@app.get("/validation", response_class=HTMLResponse)
async def validation(request: Request):
    # In a real application, you would display validation results here
    return HTMLResponse(content="<h1>Model Validation</h1><p>Validation results and option to re-run.</p>")

@app.get("/retrain", response_class=HTMLResponse)
async def retrain(request: Request):
    # In a real application, you would have controls for retraining here
    return HTMLResponse(content="<h1>Model Retraining</h1><p>Options for retraining the model.</p>")

@app.post("/predict")
async def predict_protein(sequence: str = Form(None), fasta_file: UploadFile = File(None)):
    global model
    if model is None:
        return {"error": "Model not loaded yet. Please try again in a moment."}

    if sequence:
        try:
            processed_sequence = (sequence)
            # Assuming your model's predict method returns a result you can directly return
            prediction_result = model.predict(processed_sequence)
            return {"result": prediction_result.tolist()} # Convert numpy array to list if necessary
        except Exception as e:
            return {"error": f"Error during prediction: {e}"}

    elif fasta_file:
        content = await fasta_file.read()
        # TODO: Add logic here to:
        # 1. Parse the FASTA file content to extract sequences.
        # 2. Iterate through the extracted sequences.
        # 3. For each sequence, call preprocess_sequence().
        # 4. Call model.predict() with the processed sequence data.
        # 5. Collect and return the predictions for all sequences.
        # You might need helper functions for FASTA parsing and batch processing.
        return {"message": f"FASTA file '{fasta_file.filename}' received. Processing not fully implemented yet."} # Placeholder

    else:
        return {"error": "No sequence or FASTA file provided"}
}

@app.get("/model_structure")
async def get_model_structure():
    if model is None:
        return {"error": "Model not loaded yet."}
    # TODO: Add logic to extract model structure and parameters from the loaded model
    # Example using Keras:
    # model_summary = model.summary() # You might need to capture this output
    return {"structure": "Model structure placeholder (replace with actual)", "parameters": "Model parameters placeholder (replace with actual)"} # Placeholder

@app.get("/validation_results")
async def get_validation_results():
    # TODO: Add logic to load and return stored validation results
    # You might have a function in src/model/model_evaluation.py that handles this
    return {"validation_results": "Validation results placeholder"}

@app.post("/run_validation")
async def run_model_validation():
    # TODO: Call the evaluate_model function from src.model.model_evaluation
    # Example:
    # validation_results = evaluate_model()
    return {"message": "Validation started", "results": "Validation results placeholder"} # Return actual results or confirmation

@app.post("/retrain_model")
async def retrain_the_model():
    # TODO: Call the train_model function from src.model.model_training
    # Example:
    # training_results = train_model()
    return {"message": "Retraining started", "results": "Training results placeholder"} # Return actual results or confirmation


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)