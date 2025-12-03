#!/bin/sh

# Ensure the Python environment is set up
export PATH="$HOME/.local/bin:$PATH"
poetry install
source .venv/bin/activate

# Run the FastAPI server with Uvicorn
python -u -m uvicorn src.app:app --host 0.0.0.0 --port 8000
