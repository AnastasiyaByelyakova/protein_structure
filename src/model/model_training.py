"""
Model Training for Protein Structure Prediction
Trains deep learning models to predict 3D protein structures from sequences and features using Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import argparse
from dataclasses import dataclass, asdict, fields
from datetime import datetime
import time # For timing epochs
import yaml
from utils.config_loader import load_config, ModelConfig, PathsConfig
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set TensorFlow global policy to float32 for consistency
tf.keras.mixed_precision.set_global_policy('float32')

# --- Custom Keras Layers ---
class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer
    A simple attention mechanism to weight the input features.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        # Softmax to get weights
        a = tf.keras.backend.softmax(e, axis=1)
        # Weighted sum
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

# --- Model Definition ---
def build_model(config: Any) -> Model:
    """
    Builds the deep learning model for protein structure prediction.

    Args:
        config: A ModelConfig object containing model parameters.

    Returns:
        A compiled Keras Model.
    """
    logger.info("Building model...")

    # Input layer for the protein sequence features
    input_features = keras.Input(shape=(config.sequence_length, config.feature_dim))

    # Encoder part (e.g., a stack of Bidirectional LSTMs or GRUs)
    # This helps the model understand the sequence context.
    lstm_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=config.dropout_rate))(input_features)
    lstm_2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=config.dropout_rate))(lstm_1)

    # Attention mechanism to focus on important parts of the sequence
    attention_out = AttentionLayer()(lstm_2)

    # Dense layers to predict coordinates
    dense_1 = layers.Dense(128, activation='relu')(attention_out)
    dense_2 = layers.Dense(64, activation='relu')(dense_1)

    # The output layer predicts the 3D coordinates (x, y, z) for each residue.
    # The output shape will be (sequence_length, 3).
    # Since our attention layer has a summed output, we need to adapt the model architecture.
    # A simple approach is to use a dense layer on the attention output.
    output_coords = layers.Dense(config.sequence_length * config.coordinate_dim, activation='linear')(dense_2)

    # Reshape the output to (sequence_length, 3)
    output_coords = layers.Reshape((config.sequence_length, config.coordinate_dim))(output_coords)

    model = Model(inputs=input_features, outputs=output_coords, name=config.model_name)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    logger.info("Model built and compiled successfully.")
    model.summary()
    return model

def train_model(config: Any, data_path: Path, model_dir: Path) -> Model:
    """
    Loads data, trains the model, and saves the trained model.

    Args:
        config: A ModelConfig object with training parameters.
        data_path: Path to the processed training data.
        model_dir: Directory to save the trained model.

    Returns:
        The trained Keras Model.
    """
    logger.info("Starting model training pipeline...")

    # Load processed data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    # Build the model
    model = build_model(config)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.patience, min_delta=config.min_delta, verbose=1)
    model_checkpoint = ModelCheckpoint(
        filepath=model_dir / f"{config.model_name}_{config.model_version}.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir / f"{config.model_name}_{config.model_version}_training_history.png")
    plt.show()

    logger.info("Model training completed.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein structure prediction model.")
    parser.add_argument("--config_path", type=str, default="config", help="Path to the configuration directory.")
    args = parser.parse_args()

    try:
        # Load all necessary configurations
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

        paths_config_dict = load_config(os.path.join(config_dir, "paths.yaml"))['paths']
        database_config_dict = load_config(os.path.join(config_dir, "database.yaml"))['database']
        model_config_dict = load_config(os.path.join(config_dir, "model.yaml"))['model']

        # Create dataclass objects
        paths_config = PathsConfig(**{
            k: Path(v) if k.endswith('_dir') or k == 'base_dir' else v
            for k, v in paths_config_dict.items()
        })
        model_config = ModelConfig(**model_config_dict)

        # Ensure the model directory exists
        Path(paths_config.base_dir/paths_config.models_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring model directory exists: {Path(paths_config.models_dir).resolve()}")

        # Call the main training function
        train_model(
            config=model_config,
            data_path= paths_config.base_dir/paths_config.data_dir/paths_config.processed_data_file,
            model_dir=paths_config.base_dir/paths_config.models_dir
        )

    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading configuration files or data: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
