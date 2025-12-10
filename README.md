# Protein Structure Visualization and Analysis Web Application

PLEASE KEEP IN MIND IT's STILL WORK IN PROGRESS!

This project is a web-based application designed for visualizing and analyzing protein structures. It allows users to fetch protein data from the NCBI database, process it, and visualize it in 3D. The application also includes a machine learning model for predicting protein properties.

## Features

*   Fetches protein data from the NCBI database.
*   Processes and parses PDB files.
*   Visualizes protein structures in 3D using 3Dmol.js.
*   Includes a machine learning model for feature engineering, training, and evaluation.
*   A Flask-based backend serves the application and handles data processing.
*   A user-friendly web interface for interacting with the application.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2.  Install the dependencies using Poetry:
    ```bash
    poetry install
    ```

## Usage

1.  Run the development server:
    ```bash
    ./devserver.sh
    ```
2.  Open your web browser and navigate to `http://localhost:8080`.

## Project Structure

```
.
├── devserver.sh
├── entries.idx
├── get_js.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── data_handling
│   │   ├── __init__.py
│   │   ├── main_pipeline.py
│   │   ├── ncbi_processor.py
│   │   ├── pdb_processor.py
│   │   ├── primary_parser.py
│   │   └── tertiary_parser.py
│   ├── database_handling
│   │   ├── __init__.py
│   │   ├── database_manager.py
│   │   └── db_models.py
│   ├── model
│   │   ├── feature_engineering.py
│   │   ├── model_evaluation.py
│   │   └── model_training.py
│   └── utils
│       └── config_loader.py
├── static
│   ├── css
│   │   └── main.css
│   ├── index.html
│   └── js
│       ├── 3Dmol.js
│       ├── jquery.min.js
│       └── main.js
└── tests
    ├── __init__.py
    ├── test_feature_engineering.py
    └── test_pdb_processor.py
```

## Dependencies

The main dependencies are:
*   Flask
*   Poetry
*   jQuery
*   3Dmol.js
