
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Tuple, Any
import logging
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import yaml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config_loader import load_config, ModelConfig, PathsConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# The number of physicochemical properties for each amino acid.
# This should match the number of features defined in feature_engineering.py
BASE_FEATURE_DIM = 5

# --- Data Generator ---
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras, loading samples from individual files."""
    def __init__(self, sample_paths: List[str], base_dir: Path, batch_size: int, dim: Tuple[int, int], shuffle: bool = True):
        self.sample_paths = sample_paths
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.sample_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_sample_paths = [self.sample_paths[k] for k in indexes]
        
        X, y = self.__data_generation(batch_sample_paths)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.sample_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_sample_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generates data containing batch_size samples."""
        # The feature dimension is the second element of self.dim
        X = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y = np.empty((self.batch_size, self.dim[0], 3)) # Assuming coordinate_dim is 3

        for i, path in enumerate(batch_sample_paths):
            try:
                with open(self.base_dir / path, 'rb') as f:
                    sample = pickle.load(f)
                X[i,] = sample['X']
                y[i,] = sample['y']
            except (FileNotFoundError, pickle.UnpicklingError) as e:
                logger.error(f"Error loading sample {path}: {e}")
                # Fill with zeros if a sample is corrupted or missing
                X[i,] = np.zeros(self.dim)
                y[i,] = np.zeros((self.dim[0], 3)) 
        return X, y

# --- Custom Loss Function ---
def masked_mean_squared_error(y_true, y_pred):
    mask = tf.cast(tf.reduce_any(y_true != 0.0, axis=-1), dtype=tf.float32)
    squared_error = tf.square(y_true - y_pred)
    masked_squared_error = squared_error * tf.expand_dims(mask, -1)
    total_error = tf.reduce_sum(masked_squared_error)
    num_non_padded_elements = tf.reduce_sum(mask) * tf.cast(tf.shape(y_true)[-1], tf.float32)
    return total_error / (num_non_padded_elements + 1e-8)

# --- Model Definition ---
def build_model(config: ModelConfig, feature_dim: int) -> Model:
    logger.info(f"Building model with feature dimension: {feature_dim}")
    input_features = keras.Input(shape=(config.sequence_length, feature_dim), name="input_features")
    masked_input = layers.Masking(mask_value=0.0)(input_features)
    lstm_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=config.dropout_rate))(masked_input)
    lstm_2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=config.dropout_rate))(lstm_1)
    dense_1 = layers.TimeDistributed(layers.Dense(64, activation='relu'))(lstm_2)
    output_coords = layers.TimeDistributed(layers.Dense(config.coordinate_dim, activation='linear'), name="output_coords")(dense_1)
    model = Model(inputs=input_features, outputs=output_coords, name=config.model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss=masked_mean_squared_error, metrics=['mean_absolute_error'])
    logger.info("Model built and compiled successfully.")
    model.summary()
    return model

def train_model(model_config: ModelConfig, paths_config: PathsConfig):
    logger.info("Starting model training pipeline...")
    
    processed_dir = paths_config.base_dir / paths_config.data_dir / "processed"
    manifest_path = processed_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest file not found at {manifest_path}. Please run feature_engineering.py first.")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    sample_paths = manifest["sample_paths"]

    if not sample_paths:
        logger.error("No samples found in manifest. Please run feature_engineering.py to process the data.")
        return

    train_paths, val_paths = train_test_split(sample_paths, test_size=0.2, random_state=42)

    # Determine the actual feature dimension based on the config
    use_pca = model_config.n_components is not None and model_config.n_components < BASE_FEATURE_DIM
    feature_dim = model_config.n_components if use_pca else BASE_FEATURE_DIM

    params = {
        'dim': (model_config.sequence_length, feature_dim),
        'batch_size': model_config.batch_size,
        'base_dir': processed_dir,
    }

    training_generator = DataGenerator(train_paths, shuffle=True, **params)
    
    # Check if there are enough validation samples for at least one batch
    if len(val_paths) >= model_config.batch_size:
        validation_generator = DataGenerator(val_paths, shuffle=False, **params)
        validation_steps = len(validation_generator)
        logger.info(f"Using {len(val_paths)} samples for validation, split into {validation_steps} batches.")
    else:
        validation_generator = None
        validation_steps = None
        logger.warning(
            f"Not enough validation samples ({len(val_paths)}) for a single batch of size {model_config.batch_size}. "
            "Training will proceed without validation."
        )

    model = build_model(model_config, feature_dim)

    callbacks = [
        ModelCheckpoint(
            filepath=str(paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}.h5"), 
            save_best_only=True, 
            monitor='val_loss' if validation_generator else 'loss', # Monitor training loss if no validation
            mode='min', 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if validation_generator else 'loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6, 
            verbose=1
        )
    ]

    # Only add EarlyStopping if there is a validation set
    if validation_generator:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=model_config.patience, verbose=1))

    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=model_config.epochs,
        callbacks=callbacks,
        validation_steps=validation_steps # Explicitly set validation steps
    )

    # Plotting logic based on history contents
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = paths_config.base_dir / paths_config.models_dir / f"{model_config.model_name}_{model_config.model_version}_training_history.png"
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to {plot_path}")
    
    logger.info("Model training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein structure prediction model.")
    args = parser.parse_args()

    config_dir = Path(__file__).parent.parent / 'config'
    try:
        paths_config_dict = load_config(config_dir / "paths.yaml")['paths']
        model_config_dict = load_config(config_dir / "model.yaml")['model']

        paths_config = PathsConfig(**{k: Path(v) if 'dir' in k or 'file' in k else v for k, v in paths_config_dict.items()})
        model_config = ModelConfig(**model_config_dict)

        (paths_config.base_dir / paths_config.models_dir).mkdir(parents=True, exist_ok=True)

        train_model(model_config, paths_config)

    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)
