
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import logging
from pathlib import Path
import pickle
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_training import DataGenerator, masked_mean_squared_error
from utils.config_loader import load_config, ModelConfig, PathsConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# This should match the value in model_training.py and feature_engineering.py
BASE_FEATURE_DIM = 5

class ModelEvaluator:
    """Evaluates a trained Keras model for protein structure prediction using a data generator."""
    def __init__(self, config: ModelConfig, model_path: Path):
        self.config = config
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        logger.info(f"Loading model from {self.model_path}...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Load the model with custom object scope for the custom loss function.
        with keras.utils.custom_object_scope({'masked_mean_squared_error': masked_mean_squared_error}):
            model = keras.models.load_model(
                self.model_path,
                compile=False 
            )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=masked_mean_squared_error,
            metrics=['mean_absolute_error']
        )
        logger.info("Model loaded and re-compiled successfully.")
        return model

    def evaluate(self, test_generator: DataGenerator):
        logger.info(f"Running evaluation on {len(test_generator.sample_paths)} samples...")

        # Use the generator for predictions
        y_pred = self.model.predict(test_generator, verbose=1)

        # We need the ground truth data, so let's iterate through the generator to get it
        y_true = []
        for i in tqdm(range(len(test_generator)), desc="Fetching ground truth data"):
            _, y_batch = test_generator[i]
            y_true.extend(y_batch)
        y_true = np.array(y_true)
        
        # The number of samples might not be a perfect multiple of the batch size
        num_samples = y_pred.shape[0]
        y_true = y_true[:num_samples]

        rmsd_list, gdt_ts_list, dist_mae_list = [], [], []
        raw_data = []

        for i in tqdm(range(num_samples), desc="Calculating metrics"):
            true_coords = y_true[i]
            pred_coords = y_pred[i]

            # Filter out padded zeros
            mask = np.any(true_coords != 0, axis=1)
            true_coords_filtered = true_coords[mask]
            pred_coords_filtered = pred_coords[mask]

            if len(true_coords_filtered) > 1:
                rmsd = np.sqrt(np.mean(np.sum((true_coords_filtered - pred_coords_filtered)**2, axis=1)))
                rmsd_list.append(rmsd)
                
                distances = np.linalg.norm(true_coords_filtered - pred_coords_filtered, axis=1)
                gdt_ts = np.mean([np.sum(distances <= t) / len(distances) for t in [1.0, 2.0, 4.0, 8.0]])
                gdt_ts_list.append(gdt_ts)

                true_dist_matrix = squareform(pdist(true_coords_filtered))
                pred_dist_matrix = squareform(pdist(pred_coords_filtered))
                dist_mae_list.append(mean_absolute_error(true_dist_matrix, pred_dist_matrix))

                raw_data.append({'true_coords': true_coords_filtered, 'pred_coords': pred_coords_filtered})

        # Print average metrics
        logger.info("--- Evaluation Summary ---")
        logger.info(f"Average RMSD: {np.mean(rmsd_list):.3f} Å")
        logger.info(f"Average GDT-TS: {np.mean(gdt_ts_list):.3f}")
        logger.info(f"Average Distance MAE: {np.mean(dist_mae_list):.3f}")

        return raw_data

    def visualize_results(self, raw_data: list, output_path: Path, num_examples: int = 3):
        logger.info(f"Visualizing {num_examples} protein structures...")
        num_to_plot = min(num_examples, len(raw_data))
        if num_to_plot == 0:
            logger.warning("No data to visualize.")
            return
            
        fig = plt.figure(figsize=(num_to_plot * 7, 6))
        for i in range(num_to_plot):
            ax = fig.add_subplot(1, num_to_plot, i + 1, projection='3d')
            ax.plot(raw_data[i]['true_coords'][:, 0], raw_data[i]['true_coords'][:, 1], raw_data[i]['true_coords'][:, 2], 'o-', label='True', color='blue')
            ax.plot(raw_data[i]['pred_coords'][:, 0], raw_data[i]['pred_coords'][:, 1], raw_data[i]['pred_coords'][:, 2], 'o-', label='Predicted', color='red')
            ax.set_title(f'Structure Example {i+1}')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a protein structure prediction model.")
    args = parser.parse_args()

    config_dir = Path(__file__).parent.parent / 'config'
    try:
        paths_config_dict = load_config(config_dir / "paths.yaml")['paths']
        model_config_dict = load_config(config_dir / "model.yaml")['model']

        paths_config = PathsConfig(**{k: Path(v) if 'dir' in k or 'file' in k else v for k, v in paths_config_dict.items()})
        model_config = ModelConfig(**model_config_dict)

        processed_dir = paths_config.base_dir / paths_config.data_dir / "processed"
        manifest_path = processed_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}. Please run feature engineering first.")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        sample_paths = manifest['sample_paths']

        if len(sample_paths) > 1:
            _, test_paths = train_test_split(sample_paths, test_size=0.2, random_state=42)
        else:
            test_paths = sample_paths

        if not test_paths:
            logger.warning("No samples found for evaluation. Using all available samples.")
            test_paths = sample_paths

        # Determine the actual feature dimension, same as in the training script
        use_pca = model_config.n_components is not None and model_config.n_components < BASE_FEATURE_DIM
        feature_dim = model_config.n_components if use_pca else BASE_FEATURE_DIM

        test_params = {
            'dim': (model_config.sequence_length, feature_dim),
            'batch_size': 1,  # Always use a batch size of 1 for evaluation
            'base_dir': processed_dir,
            'shuffle': False # No need to shuffle for evaluation
        }
        test_generator = DataGenerator(test_paths, **test_params)

        model_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.keras"

        evaluator = ModelEvaluator(model_config, model_path)
        raw_data = evaluator.evaluate(test_generator)
        
        if raw_data:
            output_plot_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}_evaluation_visuals.png"
            evaluator.visualize_results(raw_data, output_plot_path)
        else:
            logger.warning("No data was generated for visualization.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
