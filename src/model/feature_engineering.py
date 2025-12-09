
"""
Feature Engineering for Protein Structure Prediction
Extracts and engineers features from protein sequences and structures for model training.
This version is optimized for large datasets to avoid out-of-memory errors.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import pickle
import argparse
import os
import yaml
import sys
from tqdm import tqdm
import json

# Add the project root to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database_handling.database_manager import DatabaseManager
from utils.config_loader import load_config, ModelConfig, PathsConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mappings and Constants ---
THREE_TO_ONE_LETTER_AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XLE': 'J', 'XAA': 'X', 'UNK': 'X'
}

AMINO_ACID_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'polarity': 8.1, 'volume': 67, 'mass': 71.0779, 'charge': 0},
    'R': {'hydrophobicity': -4.5, 'polarity': 10.5, 'volume': 148, 'mass': 156.1872, 'charge': 1},
    'N': {'hydrophobicity': -3.5, 'polarity': 11.6, 'volume': 96, 'mass': 114.1026, 'charge': 0},
    'D': {'hydrophobicity': -3.5, 'polarity': 13.0, 'volume': 91, 'mass': 115.0874, 'charge': -1},
    'C': {'hydrophobicity': 2.5, 'polarity': 5.5, 'volume': 86, 'mass': 103.1429, 'charge': 0},
    'Q': {'hydrophobicity': -3.5, 'polarity': 10.5, 'volume': 114, 'mass': 128.1292, 'charge': 0},
    'E': {'hydrophobicity': -3.5, 'polarity': 12.3, 'volume': 109, 'mass': 129.114, 'charge': -1},
    'G': {'hydrophobicity': -0.4, 'polarity': 7.9, 'volume': 48, 'mass': 57.0513, 'charge': 0},
    'H': {'hydrophobicity': -3.2, 'polarity': 10.4, 'volume': 118, 'mass': 137.1393, 'charge': 1},
    'I': {'hydrophobicity': 4.5, 'polarity': 5.2, 'volume': 124, 'mass': 113.1576, 'charge': 0},
    'L': {'hydrophobicity': 3.8, 'polarity': 4.9, 'volume': 124, 'mass': 113.1576, 'charge': 0},
    'K': {'hydrophobicity': -3.9, 'polarity': 11.3, 'volume': 135, 'mass': 128.1723, 'charge': 1},
    'M': {'hydrophobicity': 1.9, 'polarity': 5.7, 'volume': 124, 'mass': 131.1961, 'charge': 0},
    'F': {'hydrophobicity': 2.8, 'polarity': 5.0, 'volume': 151, 'mass': 147.1739, 'charge': 0},
    'P': {'hydrophobicity': -1.6, 'polarity': 8.0, 'volume': 90, 'mass': 97.1152, 'charge': 0},
    'S': {'hydrophobicity': -0.8, 'polarity': 9.2, 'volume': 73, 'mass': 87.0773, 'charge': 0},
    'T': {'hydrophobicity': -0.7, 'polarity': 8.6, 'volume': 93, 'mass': 101.1039, 'charge': 0},
    'W': {'hydrophobicity': -0.9, 'polarity': 5.4, 'volume': 189, 'mass': 186.2099, 'charge': 0},
    'Y': {'hydrophobicity': -1.3, 'polarity': 6.2, 'volume': 161, 'mass': 163.1733, 'charge': 0},
    'V': {'hydrophobicity': 4.2, 'polarity': 5.9, 'volume': 105, 'mass': 99.131, 'charge': 0},
    'B': {'hydrophobicity': -3.5, 'polarity': 12.3, 'volume': 91, 'mass': 114.595, 'charge': -0.5}, # Avg of D, N
    'Z': {'hydrophobicity': -3.5, 'polarity': 11.4, 'volume': 111, 'mass': 128.6216, 'charge': -0.5}, # Avg of E, Q
    'J': {'hydrophobicity': 4.15, 'polarity': 5.05, 'volume': 124, 'mass': 113.1576, 'charge': 0}, # Avg of I, L
    'X': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0}, # Unknown
}
FEATURE_NAMES = list(AMINO_ACID_PROPERTIES['A'].keys())

class FeatureEngineer:
    """
    Handles memory-efficient feature extraction, normalization, and dimensionality reduction.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    def extract_raw_features(self, protein_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts raw features (X) and targets (y) from a single protein data entry.
        This version derives the sequence from the structure to ensure alignment.
        """
        structure_data = protein_data.get('structure_data', {})
        if not structure_data:
            return None, None

        features = []
        coords = []

        # Sort residues by residue number to maintain sequence order
        all_residues = sorted(
            [res for chain in structure_data.get('chains', {}).values() for res in chain],
            key=lambda r: r['residue_number']
        )

        for residue in all_residues:
            # Find the C-alpha atom, which is the anchor for each residue
            ca_atom = next((atom for atom in residue['atoms'] if atom['atom_name'] == 'CA'), None)
            
            if ca_atom:
                # 1. Add coordinates to the target list
                coords.append([ca_atom['x'], ca_atom['y'], ca_atom['z']])
                
                # 2. Generate features for the corresponding amino acid
                three_letter_code = residue.get('residue_name', 'UNK')
                one_letter_code = THREE_TO_ONE_LETTER_AA.get(three_letter_code, 'X')
                properties = AMINO_ACID_PROPERTIES.get(one_letter_code, AMINO_ACID_PROPERTIES['X'])
                features.append(list(properties.values()))

        # If no valid residues with C-alpha atoms were found, skip this protein
        if not coords or not features:
            return None, None

        # By construction, len(features) will now equal len(coords)
        X = np.array(features, dtype=np.float32)
        y = np.array(coords, dtype=np.float32)
            
        return X, y

    def pad_and_truncate(self, data: np.ndarray, max_len: int, dim: int) -> np.ndarray:
        """Pads or truncates a sequence to a fixed length."""
        current_len = data.shape[0]
        if current_len > max_len:
            return data[:max_len]
        elif current_len < max_len:
            padding = np.zeros((max_len - current_len, dim), dtype=data.dtype)
            return np.vstack([data, padding])
        return data

def run_feature_engineering(db_manager: DatabaseManager, paths_config: PathsConfig, model_config: ModelConfig):
    """Main function to run the memory-efficient feature engineering pipeline."""
    logger.info("--- Starting Memory-Efficient Feature Engineering Pipeline ---")

    processed_dir = paths_config.base_dir / paths_config.data_dir / "processed"
    samples_dir = processed_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processed data will be saved in: {processed_dir}")

    proteins = db_manager.get_all_proteins()
    if not proteins:
        logger.error("No proteins found in the database. Please run the main data processing pipeline first.")
        return

    feature_engineer = FeatureEngineer(config=model_config)
    
    # --- First Pass: Fit Scaler and PCA ---
    logger.info("--- Pass 1: Fitting Scaler and PCA model ---")
    scaler = StandardScaler()
    use_pca = model_config.n_components is not None and model_config.n_components < len(FEATURE_NAMES)
    pca = IncrementalPCA(n_components=model_config.n_components, batch_size=model_config.batch_size) if use_pca else None

    for protein in tqdm(proteins, desc="Fitting models"):
        X_raw, _ = feature_engineer.extract_raw_features(protein)
        if X_raw is not None and X_raw.shape[0] > 0:
            scaler.partial_fit(X_raw)
            if use_pca:
                X_scaled = scaler.transform(X_raw) # Transform is needed for PCA fitting here
                pca.partial_fit(X_scaled)
    
    with open(processed_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    if use_pca: 
        with open(processed_dir / "pca.pkl", "wb") as f: 
            pickle.dump(pca, f)
    logger.info("Scaler and PCA models have been fitted and saved.")

    # --- Second Pass: Transform and Save Individual Samples ---
    logger.info("--- Pass 2: Transforming and Saving Individual Samples ---")
    manifest = {"sample_paths": []}
    feature_dim = model_config.n_components if use_pca else len(FEATURE_NAMES)

    for i, protein in enumerate(tqdm(proteins, desc="Processing and saving samples")):
        X_raw, y_raw = feature_engineer.extract_raw_features(protein)

        if X_raw is None or y_raw is None: continue

        X_scaled = scaler.transform(X_raw)
        X_reduced = pca.transform(X_scaled) if use_pca else X_scaled
        
        X_final = feature_engineer.pad_and_truncate(X_reduced, model_config.sequence_length, feature_dim)
        y_final = feature_engineer.pad_and_truncate(y_raw, model_config.sequence_length, model_config.coordinate_dim)
        
        sample_path = samples_dir / f"sample_{i:06d}.pkl"
        with open(sample_path, "wb") as f: pickle.dump({'X': X_final, 'y': y_final}, f)
        
        manifest["sample_paths"].append(str(sample_path.relative_to(processed_dir)))

    with open(processed_dir / "manifest.json", "w") as f: json.dump(manifest, f, indent=4)
        
    logger.info(f"{len(manifest['sample_paths'])} samples were processed and saved.")
    logger.info("--- Feature Engineering Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the feature engineering pipeline.")
    args = parser.parse_args()

    config_dir = Path(__file__).parent.parent / 'config'
    try:
        paths_config_dict = load_config(config_dir / "paths.yaml")['paths']
        database_config_dict = load_config(config_dir / "database.yaml")['database']
        model_config_dict = load_config(config_dir / "model.yaml")['model']

        paths_config = PathsConfig(**{k: Path(v) if 'dir' in k or 'file' in k else v for k, v in paths_config_dict.items()})
        model_config = ModelConfig(**model_config_dict)

        db_manager = DatabaseManager(db_url=database_config_dict['database_url'])

        run_feature_engineering(db_manager, paths_config, model_config)

    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)
