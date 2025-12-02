"""
Feature Engineering for Protein Structure Prediction
Extracts and engineers features from protein sequences and structures for model training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import itertools
import pickle  # For saving/loading combined data
import argparse  # For command-line arguments in main
import os
import yaml
import sys
from tqdm import tqdm
# Add the project root to the sys.path to resolve module imports
# This is a common practice for running scripts from subdirectories
# Assuming the script is in 'feature_engineering.py' at the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary components from your project
from database_handling.database_manager import DatabaseManager
from utils.config_loader import load_config
from utils.config_loader import load_config, ModelConfig, PathsConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# A simple mapping for three-letter to one-letter amino acid codes
THREE_TO_ONE_LETTER_AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Common modifications
    'ASX': 'B', 'GLX': 'Z', 'XLE': 'J', 'XAA': 'X'
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
    # Handle unknown amino acids
    'X': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0},
    'Z': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0},
    'J': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0},
    'O': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0},
    'U': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0},
    'B': {'hydrophobicity': 0, 'polarity': 0, 'volume': 0, 'mass': 0, 'charge': 0}
}


def model_to_dict(instance):
    return {c.name: getattr(instance, c.name) for c in instance.__table__.columns}

class FeatureEngineer:
    """
    Handles feature extraction, normalization, and dimensionality reduction.
    """
    def __init__(self, config):
        """
        Initializes the feature engineer with model configuration.

        Args:
            config: An instance of ModelConfig.
            pdb_processor: An instance of PDBProcessor.
        """
        self.config = config
        self.scaler = StandardScaler()
        self.pca_model = PCA(n_components=20)
        self.feature_names = []

    def extract_features(self, protein_data: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extracts features from a single protein data entry.

        Args:
            protein_data: A dictionary containing protein information from the database.

        Returns:
            A tuple containing:
            - A NumPy array of features (X).
            - A NumPy array of target coordinates (y).
            - The original gene sequence string.
        """
        sequence = protein_data.get('gene_sequence')
        structure_data = protein_data.get('structure_data', {})

        if not sequence or not structure_data:
            logger.warning(f"Skipping protein {protein_data.get('pdb_id')} due to missing sequence or structure data.")
            return None, None, None

        # Process sequence-based features
        sequence_features = self._sequence_to_features(sequence)

        # Process structure-based targets
        target_coords, target_mask = self._structure_to_targets(structure_data)

        # Pad or truncate features and targets to a fixed length
        padded_sequence_features = self._pad_or_truncate(sequence_features, self.config.sequence_length, sequence_features.shape[-1])
        padded_target_coords = self._pad_or_truncate(target_coords, self.config.sequence_length, self.config.coordinate_dim)
        padded_target_mask = self._pad_or_truncate(target_mask, self.config.sequence_length, 1)

        # Mask out coordinates of padded residues
        padded_target_coords[padded_target_mask[:, 0] == 0] = 0

        # Update the feature dimension in the config
        self.config.feature_dim = padded_sequence_features.shape[-1]
        self.feature_names = list(AMINO_ACID_PROPERTIES['A'].keys())

        return padded_sequence_features, padded_target_coords, sequence

    def _sequence_to_features(self, sequence: str) -> np.ndarray:
        """
        Converts a protein sequence into a feature matrix using amino acid properties.
        """
        features = []
        for amino_acid in sequence:
            properties = AMINO_ACID_PROPERTIES.get(amino_acid.upper(), None)
            if properties:
                features.append(list(properties.values()))
            else:
                # Handle unknown amino acids by using a vector of zeros
                logger.warning(f"Unknown amino acid '{amino_acid}' found in sequence.")
                features.append([0] * len(list(AMINO_ACID_PROPERTIES['A'].keys())))
        return np.array(features, dtype=np.float32)

    def _structure_to_targets(self, structure_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts alpha-carbon (C-alpha) coordinates as targets.
        """
        coords = []
        # Flatten chains and residues to get a single list of residues
        all_residues = []
        for chain_id, residues_list in structure_data.get('chains', {}).items():
            all_residues.extend(residues_list)
        # Sort residues by residue number to ensure correct order
        all_residues.sort(key=lambda r: r['residue_number'])
        

        # Find the C-alpha atom for each residue
        for residue in all_residues:
            ca_atom = next((atom for atom in residue['atoms'] if atom['atom_name'] == 'CA'), None)
            if ca_atom:
                coords.append([ca_atom['x'], ca_atom['y'], ca_atom['z']])

        coords = np.array(coords, dtype=np.float32)
        # Create a mask to indicate which positions are real residues (1) and which are padding (0)
        mask = np.ones((len(coords), 1), dtype=np.float32)

        return coords, mask

    def _pad_or_truncate(self, data: np.ndarray, max_len: int, dim: int) -> np.ndarray:
        """
        Pads or truncates a sequence of data to a fixed length.
        """
        current_len = data.shape[0]
        if current_len > max_len:
            return data[:max_len]
        elif current_len < max_len:
            padding = np.zeros((max_len - current_len, dim), dtype=data.dtype)
            return np.vstack([data, padding])
        else:
            return data

    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalizes the feature matrix.

        Args:
            X: The feature matrix.
            fit: If True, fits the scaler on X before transforming.

        Returns:
            The normalized feature matrix.
        """
        # Reshape X for StandardScaler (which expects 2D data)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        print("original_shape",original_shape)
        print("-1, X.shape[-1]",[-1, X.shape[-1]])
        print("X",X)
        print("X_reshaped",X_reshaped)
        print("fit",fit)
        if fit:
            self.scaler.fit(X_reshaped)
        X_normalized_reshaped = self.scaler.transform(X_reshaped)
        print("X_normalized_reshaped",X_normalized_reshaped)
        print("X_normalized_reshaped.reshape(original_shape)",X_normalized_reshaped.reshape(original_shape))
        return X_normalized_reshaped.reshape(original_shape)

    def reduce_dimensionality(self, X: np.ndarray, n_components: int, fit: bool = False) -> np.ndarray:
        """
        Reduces the dimensionality of the feature matrix using PCA.
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        if fit:
            self.pca_model = PCA(n_components=n_components)
            self.pca_model.fit(X_reshaped)
        if self.pca_model:
            X_reduced_reshaped = self.pca_model.transform(X_reshaped)
            return X_reduced_reshaped.reshape(original_shape[0], original_shape[1], n_components)
        else:
            logger.warning("PCA model not fitted. Returning original features.")
            return X

    def save_processed_data(self, X, y, feature_names, scaler, pca_model, filepath: Path):
        """
        Saves the processed data, scaler, and PCA model to a file.
        """
        data = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'scaler': scaler,
            'pca_model': pca_model
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Processed data saved to {filepath}")

    def load_processed_data(self, filepath: Path):
        """
        Loads the processed data from a file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data.get('scaler')
        self.pca_model = data.get('pca_model')
        self.feature_names = data.get('feature_names')
        return data.get('X'), data.get('y')

    def visualize_features(self, X: np.ndarray, feature_names: List[str]):
        """
        Creates a heatmap of the feature correlation matrix.
        """
        if len(X.shape) > 2:
            X = X.reshape(-1, X.shape[-1])

        df = pd.DataFrame(X, columns=feature_names)

        # Plotting a correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

def run_feature_engineering(db_manager: DatabaseManager, paths_config: PathsConfig, model_config: ModelConfig):
    """
    Main function to run the feature engineering pipeline.
    """
    logger.info("--- Starting Feature Engineering Pipeline ---")

    # Fetch proteins from the database
    proteins = db_manager.get_all_proteins()
    if not proteins:
        logger.error("No proteins found in the database. Please run the main pipeline first.")
        return

    # Initialize FeatureEngineer
    feature_engineer = FeatureEngineer(config=model_config)

    # Process all proteins and collect features and targets
    all_features = []
    all_targets = []
    for protein in tqdm(proteins, desc="Processing proteins"):
        X_protein, y_protein, _ = feature_engineer.extract_features(protein)
        if X_protein is not None and y_protein is not None:
            all_features.append(X_protein)
            all_targets.append(y_protein)

    if not all_features:
        logger.error("No valid proteins were processed. Exiting.")
        return

    # Stack all features and targets into single arrays
    X = np.stack(all_features)
    y = np.stack(all_targets)

    logger.info(f"Initial data shape: X={X.shape}, y={y.shape}")

    # Normalize features
    logger.info("Normalizing features...")
    X_normalized = feature_engineer.normalize_features(X, fit=True)

    # Store feature names after creation
    feature_names = feature_engineer.feature_names

    # Reduce dimensionality
    if model_config.n_components is not None and model_config.n_components < X.shape[-1]:
        logger.info(f"Reducing dimensionality to {model_config.n_components} components...")
        X_reduced = feature_engineer.reduce_dimensionality(X_normalized, n_components=model_config.n_components, fit=True)
    else:
        X_reduced = X_normalized
        logger.info("Skipping dimensionality reduction.")

    # Save all processed data to a single pickle file as a dictionary
    output_path = paths_config.base_dir/paths_config.data_dir/paths_config.processed_data_file
    feature_engineer.save_processed_data(
        X=X_reduced,
        y=y,
        feature_names=feature_names, # Use the stored feature names
        scaler=feature_engineer.scaler,
        pca_model=feature_engineer.pca_model,
        filepath=output_path
    )

    # Create visualizations (optional)
    logger.info("Creating visualizations...")
    feature_engineer.visualize_features(X_reduced.reshape(-1, X_reduced.shape[-1]), feature_engineer.feature_names)

    # Print summary
    print(f"\nFeature Engineering Summary:")
    print(f"Total proteins processed: {len(proteins)}")
    print(f"Original features (before PCA): {X_normalized.shape[-1]}")
    print(f"Reduced features (after PCA): {X_reduced.shape[-1]}")
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run the feature engineering pipeline.")
        args = parser.parse_args()

        # Load configurations
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

    # try:
        paths_config_dict = load_config(os.path.join(config_dir, "paths.yaml"))['paths']
        database_config_dict = load_config(os.path.join(config_dir, "database.yaml"))['database']
        model_config_dict = load_config(os.path.join(config_dir, "model.yaml"))['model']

        # Create dataclass objects
        paths_config = PathsConfig(**{
            k: Path(v) if k.endswith('_dir') or k == 'base_dir' else v
            for k, v in paths_config_dict.items()
        })
        model_config = ModelConfig(**model_config_dict)

        # Initialize DatabaseManager
        db_manager = DatabaseManager(db_url=database_config_dict['database_url'])

        # Run the main function with the new objects
        run_feature_engineering(db_manager, paths_config, model_config)
