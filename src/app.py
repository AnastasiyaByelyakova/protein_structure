import os
import io
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
from contextlib import asynccontextmanager
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
import yaml
import pickle
import numpy as np
import copy
from dataclasses import asdict
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from src.model.model_training import train_model, AttentionLayer
from src.model.model_evaluation import ModelEvaluator
from src.model.feature_engineering import FeatureEngineer, AMINO_ACID_PROPERTIES
from utils.config_loader import PathsConfig, ModelConfig, DatabaseConfig, load_config
from database_handling.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Globals ---
model = None
model_config = None
paths_config = None
db_manager = None
feature_engineer = None
last_validation_results = None

# --- Security & Auth ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---
class User(BaseModel):
    password: str
    email: str

class UserInDB(User):
    hashed_password: str

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

# --- FastAPI Lifespan (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_config, paths_config, db_manager, feature_engineer, last_validation_results
    logger.info("Server startup: Loading configurations and initializing components...")
    try:
        CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
        paths_config_dict = load_config(os.path.join(CONFIG_DIR, "paths.yaml"))['paths']
        database_config_dict = load_config(os.path.join(CONFIG_DIR, "database.yaml"))['database']
        model_config_dict = load_config(os.path.join(CONFIG_DIR, "model.yaml"))['model']

        paths_config = PathsConfig(**{k: Path(v) for k, v in paths_config_dict.items()})
        model_config = ModelConfig(**model_config_dict)
        db_config = DatabaseConfig(**database_config_dict)

        db_manager = DatabaseManager(db_url=db_config.database_url)
        feature_engineer = FeatureEngineer(config=model_config)

        processed_data_path = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file
        if processed_data_path.exists():
            logger.info(f"Loading processed data from {processed_data_path} to fit scaler and PCA...")
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            feature_engineer.scaler = data.get('scaler')
            feature_engineer.pca_model = data.get('pca_model')
            if feature_engineer.scaler and feature_engineer.pca_model:
                logger.info("Scaler and PCA models loaded for FeatureEngineer.")
            else:
                logger.warning("Scaler or PCA model not found in processed data. Feature engineering might be incomplete.")

        model_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.h5"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}...")
            model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model file not found at {model_path}. Prediction endpoint will not be available.")
            model = None

        logger.info("Server startup complete.")
        yield
    except Exception as e:
        logger.error(f"Failed to load configurations or initialize components during startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Server shutdown.")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "static"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# --- Database & User Management ---
def get_db():
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not initialized.")
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, username: str):
    return db_manager.get_user_by_username(db, username)

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    user = get_user(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials", headers={"WWW-Authenticate": "Bearer"})
    return user

# --- Web Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Authentication Endpoints ---
@app.post("/register/")
async def register_user(user: User, db: Session = Depends(get_db)):
    if get_user(db, user.email):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_manager.create_user(db, user.email, hashed_password)
    return {"message": "User registered successfully"}

@app.post("/login/")
async def login_user(user: User, db: Session = Depends(get_db)):
    user_in_db = get_user(db, user.email)
    if not user_in_db or not verify_password(user.password, user_in_db.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    return {"message": "Login successful", "access_token": user.email, "token_type": "bearer"}

# --- Core Functionality Endpoints ---
def is_valid_protein_sequence(sequence: str) -> bool:
    valid_chars = set(AMINO_ACID_PROPERTIES.keys())
    return sequence and all(char.upper() in valid_chars for char in sequence)

@app.post("/predict/", dependencies=[Depends(get_current_user)])
async def predict_protein(sequence: str = Form(None), fasta_file: UploadFile = File(None)):
    if not model or not feature_engineer or not model_config:
        raise HTTPException(status_code=503, detail="Model or components not initialized.")

    # This endpoint logic is complex and would be refactored for production.
    # For now, it demonstrates the prediction flow.
    # (Simplified prediction logic as before)
    if sequence:
        if not is_valid_protein_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence.")
        sequences_to_predict = {'input_sequence': sequence}
    else:
        raise HTTPException(status_code=400, detail="No sequence provided.")

    try:
        # Simplified feature extraction for a single sequence
        features = feature_engineer._sequence_to_features(sequences_to_predict['input_sequence'])
        print(features)
        print(feature_engineer)
        padded = feature_engineer._pad_or_truncate(features, model_config.sequence_length, features.shape[1])
        normalized = feature_engineer.normalize_features(padded, fit=False)
        print(feature_engineer.pca_model)
        reduced = feature_engineer.pca_model.transform(normalized)
        X_predict = np.expand_dims(reduced, axis=0)
        prediction = model.predict(X_predict)
        return JSONResponse(content={"predictions": prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


@app.get("/model_info/", dependencies=[Depends(get_current_user)])
async def get_model_info():
    if not model or not model_config:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    model_summary = stream.getvalue()

    info = {
        "model_name": model_config.model_name,
        "model_version": model_config.model_version,
        "model_summary": model_summary,
        "last_validation_results": "No recent validation results available."
    }
    if last_validation_results:
        info["last_validation_results"] = convert_numpy_types(asdict(last_validation_results))

    return JSONResponse(content=info)

@app.get("/validation/", dependencies=[Depends(get_current_user)])
async def get_validation_status():
    if last_validation_results:
        return JSONResponse(content=convert_numpy_types(asdict(last_validation_results)))
    return JSONResponse(content={"status": "No validation has been run."})

@app.post("/run_validation/", dependencies=[Depends(get_current_user)])
async def trigger_validation():
    global last_validation_results
    if not all([model, model_config, paths_config, feature_engineer]):
        raise HTTPException(status_code=503, detail="Server components not initialized.")

    logger.info("Triggering model validation...")
    try:
        processed_data_path = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file
        if not processed_data_path.exists():
            raise HTTPException(status_code=400, detail="Processed data file not found.")

        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        _, X_val, _, y_val = train_test_split(data['X'], data['y'], test_size=0.1, random_state=42)
        
        model_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.h5"
        evaluator = ModelEvaluator(model_config, str(model_path))
        metrics, _ = evaluator.evaluate((X_val, y_val))
        last_validation_results = metrics

        logger.info("Validation complete.")
        return JSONResponse(content={
            "message": "Validation completed successfully.",
            "results": convert_numpy_types(asdict(metrics))
        })
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during validation: {e}")

@app.post("/retrain_model/", dependencies=[Depends(get_current_user)])
async def trigger_retraining(
    epochs: int = Form(...),
    batch_size: int = Form(...),
    learning_rate: float = Form(...),
    new_dataset: Optional[UploadFile] = File(None)
):
    global model, model_config, paths_config
    if not all([model_config, paths_config]):
        raise HTTPException(status_code=503, detail="Server components not initialized.")

    logger.info(f"Triggering model retraining with params: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    try:
        temp_config = copy.deepcopy(model_config)
        temp_config.epochs = epochs
        temp_config.batch_size = batch_size
        temp_config.learning_rate = learning_rate

        data_path_to_use = paths_config.base_dir / paths_config.data_dir / paths_config.processed_data_file

        if new_dataset and new_dataset.filename:
            logger.info(f"Processing uploaded dataset: {new_dataset.filename}")
            
            content = await new_dataset.read()
            if not content:
                raise HTTPException(status_code=400, detail="The uploaded dataset file is empty.")

            uploaded_data_path = paths_config.base_dir / paths_config.data_dir / "uploaded_processed_data.pkl"
            with open(uploaded_data_path, 'wb') as f:
                f.write(content)
            data_path_to_use = uploaded_data_path
            logger.info(f"Using uploaded dataset from {data_path_to_use}")

        if not data_path_to_use.exists():
            raise HTTPException(status_code=400, detail=f"Data file not found at {data_path_to_use}.")

        trained_model = train_model(
            config=temp_config,
            data_path=str(data_path_to_use),
            model_dir=str(paths_config.base_dir / paths_config.models_dir)
        )

        model = trained_model
        logger.info("Model retraining completed. New model is now loaded.")
        
        return JSONResponse(content={
            "message": "Model retraining completed successfully. The new model is now active.",
            "new_config": asdict(temp_config)
        })

    except Exception as e:
        logger.error(f"Error during retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during retraining: {e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    try:
        app_config = load_config(os.path.join(os.path.dirname(__file__), 'config', "app.yaml"))['app']
        host = app_config.get("host", "0.0.0.0")
        port = app_config.get("port", 8000)
    except Exception:
        host, port = "0.0.0.0", 8000
    
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
