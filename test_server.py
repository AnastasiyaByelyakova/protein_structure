
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
