"""
Model Evaluation for Protein Structure Prediction
Comprehensive evaluation of trained models including structural metrics and visualizations (Keras version)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import os

# Import the entire model_training module to ensure custom layers are in scope
from src.model.model_training import *
from src.utils.config_loader import load_config, ModelConfig, PathsConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Coordinate-based metrics
    rmsd: float
    gdt_ts: float
    gdt_ha: float
    # Distance-based metrics
    distance_mae: float
    distance_rmse: float
    distance_correlation: float
    # Contact-based metrics
    contact_precision_5A: float
    contact_recall_5A: float
    contact_f1_5A: float

class ModelEvaluator:
    """
    Evaluates a trained Keras model for protein structure prediction.
    """
    def __init__(self, config: Any, model_path: Path):
        """
        Initializes the evaluator with the model and configuration.

        Args:
            config: An object containing model configurations.
            model_path: The path to the trained Keras model file (.h5).
        """
        self.config = config
        self.model_path = model_path
        self.model = self._load_model()
        logger.info(f"Model {self.config.model_name} loaded from {self.model_path}")

    def _load_model(self):
        """Loads the saved Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Need to pass the custom layers to the load_model function
        return keras.models.load_model(
            self.model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )

    def _calculate_rmsd(self, true_coords: np.ndarray, pred_coords: np.ndarray) -> float:
        """
        Calculates Root Mean Square Deviation (RMSD).

        Args:
            true_coords: True coordinates, shape (N, 3).
            pred_coords: Predicted coordinates, shape (N, 3).

        Returns:
            The RMSD value in Angstroms.
        """
        # Ensure coordinates have the same shape
        if true_coords.shape != pred_coords.shape:
            raise ValueError("Coordinate arrays must have the same shape.")

        # Calculate squared differences
        diff = true_coords - pred_coords
        sq_diff = diff**2

        # Calculate mean squared difference and take the square root
        rmsd = np.sqrt(np.mean(sq_diff))
        return rmsd

    def _calculate_gdt(self, true_coords: np.ndarray, pred_coords: np.ndarray) -> Tuple[float, float]:
        """
        Calculates Global Distance Test (GDT) metrics.
        This is a simplified version for illustration.

        Args:
            true_coords: True coordinates, shape (N, 3).
            pred_coords: Predicted coordinates, shape (N, 3).

        Returns:
            A tuple (gdt_ts, gdt_ha).
        """
        distances = np.linalg.norm(true_coords - pred_coords, axis=1)

        # GDT-TS (Total Score) thresholds
        thresholds_ts = [1.0, 2.0, 4.0, 8.0]
        gdt_ts_scores = [np.sum(distances <= t) / len(distances) for t in thresholds_ts]
        gdt_ts = np.mean(gdt_ts_scores)

        # GDT-HA (High Accuracy) thresholds
        thresholds_ha = [0.5, 1.0, 2.0, 4.0]
        gdt_ha_scores = [np.sum(distances <= t) / len(distances) for t in thresholds_ha]
        gdt_ha = np.mean(gdt_ha_scores)

        return gdt_ts, gdt_ha

    def _calculate_contact_metrics(self, true_coords: np.ndarray, pred_coords: np.ndarray, threshold: float = 5.0):
        """
        Calculates contact map metrics (Precision, Recall, F1-score).

        Args:
            true_coords: True coordinates, shape (N, 3).
            pred_coords: Predicted coordinates, shape (N, 3).
            threshold: Distance threshold for a contact (in Angstroms).

        Returns:
            A tuple (precision, recall, f1).
        """
        # Calculate distance matrices
        true_dist_matrix = squareform(pdist(true_coords))
        pred_dist_matrix = squareform(pdist(pred_coords))

        # Create contact maps
        true_contacts = (true_dist_matrix < threshold).astype(int)
        pred_contacts = (pred_dist_matrix < threshold).astype(int)

        # Count true positives, false positives, false negatives
        tp = np.sum(true_contacts * pred_contacts)
        fp = np.sum((1 - true_contacts) * pred_contacts)
        fn = np.sum(true_contacts * (1 - pred_contacts))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def _calculate_distance_metrics(self, true_coords: np.ndarray, pred_coords: np.ndarray):
        """
        Calculates distance-based metrics (MAE, RMSE, Correlation).

        Args:
            true_coords: True coordinates, shape (N, 3).
            pred_coords: Predicted coordinates, shape (N, 3).

        Returns:
            A tuple (mae, rmse, corr).
        """
        true_dist_matrix = squareform(pdist(true_coords))
        pred_dist_matrix = squareform(pdist(pred_coords))

        # Exclude the diagonal (self-distances)
        true_distances = true_dist_matrix[np.triu_indices(true_dist_matrix.shape[0], k=1)]
        pred_distances = pred_dist_matrix[np.triu_indices(pred_dist_matrix.shape[0], k=1)]

        mae = mean_absolute_error(true_distances, pred_distances)
        rmse = np.sqrt(mean_squared_error(true_distances, pred_distances))

        # Pearson correlation coefficient
        corr, _ = pearsonr(true_distances, pred_distances)

        return mae, rmse, corr

    def evaluate(self, test_dataset: Tuple[np.ndarray, np.ndarray]) -> Tuple[EvaluationMetrics, List[Dict[str, Any]]]:
        """
        Runs the full evaluation pipeline.

        Args:
            test_dataset: A tuple (X_test, y_test) of numpy arrays.

        Returns:
            A tuple containing an EvaluationMetrics object and raw data for visualization.
        """
        X_test, y_test = test_dataset

        logger.info(f"Running evaluation on {len(X_test)} samples...")

        # Predict coordinates for the test set
        y_pred = self.model.predict(X_test)

        # Ensure the predicted coordinates have a consistent shape
        if y_pred.shape != y_test.shape:
            # Handle potential shape mismatches, e.g., due to masking or padding
            logger.warning(f"Shape mismatch: y_pred={y_pred.shape}, y_test={y_test.shape}. "
                           "This may indicate issues with padding or model output.")

        # Initialize lists to store metrics per protein
        rmsd_list, gdt_ts_list, gdt_ha_list = [], [], []
        dist_mae_list, dist_rmse_list, dist_corr_list = [], [], []
        contact_prec_list, contact_rec_list, contact_f1_list = [], [], []

        raw_data = []

        # Iterate over each protein in the test set
        for i in tqdm(range(len(X_test)), desc="Calculating metrics per sample"):
            true_coords = y_test[i]
            pred_coords = y_pred[i]

            # Filter out padded zeros
            non_zero_indices = np.any(true_coords != 0, axis=1)
            true_coords_filtered = true_coords[non_zero_indices]
            pred_coords_filtered = pred_coords[non_zero_indices]

            # Calculate metrics if there are valid residues
            if len(true_coords_filtered) > 1: # Need at least 2 points for distance metrics
                rmsd_list.append(self._calculate_rmsd(true_coords_filtered, pred_coords_filtered))
                gdt_ts, gdt_ha = self._calculate_gdt(true_coords_filtered, pred_coords_filtered)
                gdt_ts_list.append(gdt_ts)
                gdt_ha_list.append(gdt_ha)

                mae, rmse, corr = self._calculate_distance_metrics(true_coords_filtered, pred_coords_filtered)
                dist_mae_list.append(mae)
                dist_rmse_list.append(rmse)
                dist_corr_list.append(corr)

                prec, rec, f1 = self._calculate_contact_metrics(true_coords_filtered, pred_coords_filtered)
                contact_prec_list.append(prec)
                contact_rec_list.append(rec)
                contact_f1_list.append(f1)

                raw_data.append({
                    'true_coords': true_coords_filtered,
                    'pred_coords': pred_coords_filtered,
                })

        # Calculate average metrics
        avg_metrics = EvaluationMetrics(
            rmsd=np.mean(rmsd_list),
            gdt_ts=np.mean(gdt_ts_list),
            gdt_ha=np.mean(gdt_ha_list),
            distance_mae=np.mean(dist_mae_list),
            distance_rmse=np.mean(dist_rmse_list),
            distance_correlation=np.mean(dist_corr_list),
            contact_precision_5A=np.mean(contact_prec_list),
            contact_recall_5A=np.mean(contact_rec_list),
            contact_f1_5A=np.mean(contact_f1_list)
        )

        logger.info("Evaluation complete.")
        return avg_metrics, raw_data

    def visualize_results(self, raw_data: List[Dict[str, Any]], num_examples: int = 3):
        """
        Visualizes a few predicted vs. true structures.

        Args:
            raw_data: A list of dictionaries with true and predicted coordinates.
            num_examples: Number of examples to visualize.
        """
        logger.info(f"Visualizing {num_examples} protein structures...")

        # Create a figure with subplots
        fig = plt.figure(figsize=(num_examples * 8, 8))
        for i in range(min(num_examples, len(raw_data))):
            true_coords = raw_data[i]['true_coords']
            pred_coords = raw_data[i]['pred_coords']

            # True structure plot
            ax_true = fig.add_subplot(2, num_examples, i + 1, projection='3d')
            ax_true.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2], 'o-', label='True Structure', color='blue')
            ax_true.set_title(f'True Structure (Example {i+1})')
            ax_true.set_xlabel('X')
            ax_true.set_ylabel('Y')
            ax_true.set_zlabel('Z')
            ax_true.legend()

            # Predicted structure plot
            ax_pred = fig.add_subplot(2, num_examples, i + 1 + num_examples, projection='3d')
            ax_pred.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], 'o-', label='Predicted Structure', color='red')
            ax_pred.set_title(f'Predicted Structure (Example {i+1})')
            ax_pred.set_xlabel('X')
            ax_pred.set_ylabel('Y')
            ax_pred.set_zlabel('Z')
            ax_pred.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a protein structure prediction model.")
    args = parser.parse_args()

    # Load configurations and data
    try:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

        paths_config_dict = load_config(os.path.join(config_dir, "paths.yaml"))['paths']
        model_config_dict = load_config(os.path.join(config_dir, "model.yaml"))['model']

        # Create dataclass objects
        paths_config = PathsConfig(**{
            k: Path(v) if k.endswith('_dir') or k == 'base_dir' else v
            for k, v in paths_config_dict.items()
        })
        model_config = ModelConfig(**model_config_dict)

        processed_data_path = paths_config.base_dir/ paths_config.data_dir / paths_config.processed_data_file
        if not processed_data_path.exists():
            raise FileNotFoundError(f"Processed data file not found at {processed_data_path}. Please run feature engineering first.")

        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)

        X = data['X']
        y = data['y']

        # Use a small portion of the data for testing
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        test_dataset = (X_test, y_test)

        # If no model path is provided, try to find the latest best model in the models directory
        model_path_to_use = Path(str(paths_config.base_dir/paths_config.models_dir/model_config.model_name)+'_'+model_config.model_version+'.h5')

        # Initialize evaluator
        evaluator = ModelEvaluator(model_config, model_path_to_use)

        # Evaluate the model
        metrics, raw_data = evaluator.evaluate(test_dataset)

        # Print summary metrics
        logger.info("\n--- Evaluation Summary ---")
        logger.info(f"Average RMSD: {metrics.rmsd:.3f} Å")
        logger.info(f"Average GDT-TS: {metrics.gdt_ts:.3f}")
        logger.info(f"Average GDT-HA: {metrics.gdt_ha:.3f}")
        logger.info(f"Average Distance MAE: {metrics.distance_mae:.3f}")
        logger.info(f"Average Distance RMSE: {metrics.distance_rmse:.3f}")
        logger.info(f"Average Distance Correlation (Pearson): {metrics.distance_correlation:.3f}")
        logger.info(f"Average 5Å Contact Precision: {metrics.contact_precision_5A:.3f}")
        logger.info(f"Average 5Å Contact Recall: {metrics.contact_recall_5A:.3f}")
        logger.info(f"Average 5Å Contact F1-Score: {metrics.contact_f1_5A:.3f}")

        # Create visualizations
        evaluator.visualize_results(raw_data)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except (yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading configuration files or data: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}")
