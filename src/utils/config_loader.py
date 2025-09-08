"""
Centralized Configuration for the Protein Structure Prediction Project.
This module now loads configurations from separate YAML files.
"""

import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
import yaml
import logging
from typing import *

logger = logging.getLogger(__name__)

# --- Configuration Loader Function ---

def load_config(config_path: Union[str, Path]) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise

# --- Dataclasses for Type Hinting and Structure ---
# These dataclasses are now primarily for defining the expected structure
# and are loaded from the YAML files.

@dataclass
class PathsConfig:
    """Configuration for file and directory paths."""
    base_dir: Path
    data_dir: Path
    models_dir: Path
    config_dir: Path
    templates_dir: Path
    static_dir: Path
    logs_dir: Path
    processed_data_file: str
    pdb_download_dir: Path

@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    database_url: str

@dataclass
class ModelConfig:
    """Configuration for model training."""
    sequence_length: int
    feature_dim: int
    coordinate_dim: int
    n_components: int
    learning_rate: float
    batch_size: int
    epochs: int
    dropout_rate: float
    patience: int
    min_delta: float
    model_name: str
    model_version: str
    retrain_interval_days: int
    retrain_min_samples: int
    retrain_max_attempts: int

# --- Main Configuration Loading ---
# This part is just for example usage or if you want to load a central config.
# Individual scripts can now load their specific configs directly.
# However, for a single point of truth, you can load everything here.

def get_configs(config_dir: Union[str, Path] = "config"):
    """Loads all configuration files into dataclasses."""
    config_dir = Path(config_dir)
    paths_config_dict = load_config(config_dir / "paths.yaml")['paths']
    database_config_dict = load_config(config_dir / "database.yaml")['database']
    model_config_dict = load_config(config_dir / "model.yaml")['model']

    paths_config = PathsConfig(**{
        k: Path(v) if k.endswith('_dir') or k == 'base_dir' else v
        for k, v in paths_config_dict.items()
    })
    database_config = DatabaseConfig(**database_config_dict)
    model_config = ModelConfig(**model_config_dict)

    return paths_config, database_config, model_config

if __name__ == "__main__":
    # Example usage
    try:
        paths, db, model = get_configs()
        print("Paths Config:")
        print(asdict(paths))
        print("\nDatabase Config:")
        print(asdict(db))
        print("\nModel Config:")
        print(asdict(model))
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Failed to load configs: {e}")
