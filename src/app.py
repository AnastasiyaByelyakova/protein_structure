
import os
import io
import base64
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import logging
import yaml
import pickle
import numpy as np
import copy
import pandas as pd
from dataclasses import asdict
from sklearn.model_selection import train_test_split
import google.generativeai as genai

from tensorflow.keras.models import load_model
from src.model.model_training import train_model, AttentionLayer
from src.model.model_evaluation import ModelEvaluator
from src.model.feature_engineering import FeatureEngineer, AMINO_ACID_PROPERTIES
from src.utils.config_loader import PathsConfig, ModelConfig, load_config
from src.data_handling.pdb_processor import coordinates_to_pdb

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Globals ---
model = None
model_config = None
paths_config = None
feature_engineer = None
last_validation_results = None
gemini_api_key = None

# --- Helper Functions ---
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types in a dictionary."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    return obj

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_config, paths_config, feature_engineer, last_validation_results, gemini_api_key
    logger.info("Server startup...")
    try:
        CONFIG_DIR = Path(__file__).parent / 'config'
        paths_config_dict = load_config(CONFIG_DIR / "paths.yaml")['paths']
        model_config_dict = load_config(CONFIG_DIR / "model.yaml")['model']
        credentials = load_config(CONFIG_DIR / "credentials.yaml")
        gemini_api_key = credentials.get("gemini_api_key")

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

        paths_config = PathsConfig(**{k: Path(v) for k, v in paths_config_dict.items()})
        model_config = ModelConfig(**model_config_dict)

        feature_engineer = FeatureEngineer(config=model_config)

        processed_data_path = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file
        if processed_data_path.exists():
            logger.info(f"Loading data from {processed_data_path}")
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            feature_engineer.scaler = data.get('scaler')
            feature_engineer.pca_model = data.get('pca_model')

        model_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.h5"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        else:
            logger.warning(f"Model file not found at {model_path}.")
            model = None

        logger.info("Startup complete.")
        yield
    finally:
        logger.info("Server shutdown.")

app = FastAPI(lifespan=lifespan)

STATIC_DIR = "static"
templates = Jinja2Templates(directory=STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def is_valid_protein_sequence(sequence: str) -> bool:
    return sequence and all(char.upper() in AMINO_ACID_PROPERTIES for char in sequence)

@app.post("/predict/")
async def predict_protein(sequence: str = Form(None), fasta_file: UploadFile = File(None)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    current_sequence = ''
    if sequence:
        current_sequence = sequence
    elif fasta_file:
        content = await fasta_file.read()
        lines = content.decode('utf-8').strip().split('\n')
        if not lines or not lines[0].startswith('>'):
            raise HTTPException(status_code=400, detail="Invalid FASTA format.")
        current_sequence = "".join(lines[1:])

    if not is_valid_protein_sequence(current_sequence):
        raise HTTPException(status_code=400, detail="Invalid or empty protein sequence.")

    try:
        features = feature_engineer._sequence_to_features(current_sequence)
        padded = feature_engineer._pad_or_truncate(features, model_config.sequence_length, features.shape[1])
        normalized = feature_engineer.normalize_features(padded, fit=False)
        
        X_predict = np.expand_dims(feature_engineer.pca_model.transform(normalized) if feature_engineer.pca_model else normalized, axis=0)

        prediction = model.predict(X_predict)
        prediction_list = prediction.tolist()[0]

        df = pd.DataFrame(prediction_list, columns=['x', 'y', 'z'])
        csv_data = df.to_csv(index=False)

        pdb_data = coordinates_to_pdb(prediction_list, current_sequence)

        return JSONResponse(content={"csv_data": csv_data, "pdb_data": pdb_data})

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.post("/describe_protein/")
async def describe_protein(request: Request):
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured.")

    try:
        data = await request.json()
        pdb_data = data.get('pdb_data')
        if not pdb_data:
            raise HTTPException(status_code=400, detail="PDB data is missing.")

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Provide a detailed analysis of the following protein structure, including secondary structure elements, functional annotations, and potential clinical relevance. Return the output in HTML format.\n\n{pdb_data}"
        response = model.generate_content(prompt)
        description = response.text.replace('```html','').replace('```','')
        return JSONResponse({"description":description })

    except Exception as e:
        logger.error(f"Error generating description: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {e}")

@app.get("/model_info/")
async def get_model_info():
    if not model or not model_config:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    
    return JSONResponse({
        "model_summary": stream.getvalue(),
        "last_validation_results": convert_numpy_types(asdict(last_validation_results)) if last_validation_results else None
    })

@app.get("/validation/")
async def get_validation_status():
    if not last_validation_results:
        return JSONResponse(content={"status": "No validation has been run."})
    return JSONResponse(content=convert_numpy_types(asdict(last_validation_results)))

@app.post("/run_validation/")
async def trigger_validation():
    global last_validation_results
    if not all([model, paths_config, feature_engineer]):
        raise HTTPException(status_code=503, detail="Server components not ready.")

    try:
        data_path = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Processed data not found.")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        _, X_val, _, y_val = train_test_split(data['X'], data['y'], test_size=0.1, random_state=42)

        model_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.h5"
        evaluator = ModelEvaluator(model_config, str(model_path))
        metrics, _ = evaluator.evaluate((X_val, y_val))
        last_validation_results = metrics

        return JSONResponse({"results": convert_numpy_types(asdict(metrics))})

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain_model/")
async def trigger_retraining(
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(...),
    new_dataset: Optional[UploadFile] = File(None)
):
    global model

    try:
        temp_config = copy.deepcopy(model_config)
        temp_config.epochs = epochs
        temp_config.batch_size = batch_size
        temp_config.learning_rate = learning_rate

        data_path = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file
        if new_dataset and new_dataset.filename:
            uploaded_path = paths_config.base_dir / paths_config.data_dir / "uploaded_data.pkl"
            with open(uploaded_path, "wb") as f:
                f.write(await new_dataset.read())
            data_path = uploaded_path

        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Data file not found.")

        trained_model = train_model(temp_config, str(data_path), str(paths_config.base_dir / paths_config.models_dir))
        model = trained_model

        return JSONResponse({"message": "Model retrained.", "new_config": asdict(temp_config)})

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    try:
        app_config = load_config(Path(__file__).parent / 'config' / "app.yaml")['app']
        host, port = app_config.get("host", "0.0.0.0"), app_config.get("port", 8000)
    except Exception:
        host, port = "0.0.0.0", 8000

    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
