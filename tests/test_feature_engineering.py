import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import os
import pickle
import yaml
import pathlib
import matplotlib.pyplot as plt
mock_database_manager = MagicMock()
mock_tqdm = MagicMock()

mock_scaler = MagicMock()
mock_pca = MagicMock()

try:
    from src.model.feature_engineering import FeatureEngineer, run_feature_engineering, AMINO_ACID_PROPERTIES
except ImportError:
    # If running from a different directory or using a different pytest setup,
    # you might need to adjust the import path. # type: ignore
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from model.feature_engineering import FeatureEngineer, run_feature_engineering

from src.utils.config_loader import ModelConfig, PathsConfig
import seaborn as sns


# Dummy configuration data
DUMMY_CONFIG = {
 'paths': {
    'base_dir': os.path.abspath('.'),
    "data_dir":'data',
    "models_dir":'models',
    "config_dir":"",
    "templates_dir":"templates_dir",
    "static_dir":"static_dir",
    "logs_dir":'logs',
    "pdb_download_dir":"",
    'processed_data_file': 'processed_data.pkl'},
 'database': {
 'database_url': 'sqlite:///:memory:'
 },
'model': {
        'sequence_length': 10,
        'coordinate_dim': 3,
        'n_components': 2,
        "feature_dim":"",
        "learning_rate":"",
        "batch_size":"",
        "epochs":"",        
        "dropout_rate":"",
        "patience":"",        
        "min_delta":"",
        "model_name":"",        
        "model_version":"",
        "retrain_interval_days":"",        
        "retrain_min_samples":"",
        "retrain_max_attempts":"",       
    }
}


dummy_paths_config = PathsConfig(**DUMMY_CONFIG['paths'])
dummy_model_config = ModelConfig(**DUMMY_CONFIG['model'])

# Dummy protein data from the database
DUMMY_PROTEINS = [
    {
        'pdb_id': 'protein1',
        'gene_sequence': 'ACD',
        'structure_data': {
            'chains': {
                'A': [
                    {'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 1.0, 'y': 2.0, 'z': 3.0}]},
                    {'residue_number': 2, 'atoms': [{'atom_name': 'CA', 'x': 4.0, 'y': 5.0, 'z': 6.0}]},
                    {'residue_number': 3, 'atoms': [{'atom_name': 'CA', 'x': 7.0, 'y': 8.0, 'z': 9.0}]}
                ]
            }
        }
    },
    {
        'pdb_id': 'protein2',
        'gene_sequence': 'AX',
        'structure_data': {
            'chains': {
                'A': [
                    {'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 10.0, 'y': 11.0, 'z': 12.0}]},
                    {'residue_number': 2, 'atoms': []} # Residue with no CA
                ]
            }
        }

 },
]

# Create a MagicMock instance and set attributes afterwards
mock_protein3 = MagicMock()
mock_protein3.protein_id = 'protein3' # Add protein_id for consistency, though pdb_id is used for lookup
mock_protein3.pdb_id = 'protein3' # Add pdb_id for logging
mock_protein3.gene_sequence = 'C' * 15  # Longer than sequence_length
mock_protein3.structure_data = {'CA': [{'x': i, 'y': i+1, 'z': i+2} for i in range(15)]}

DUMMY_PROTEINS.append(mock_protein3)

# Mocking the dataclass objects and the load_config function
@pytest.fixture
def mock_configs():
    """Fixture to provide mock configuration objects."""
    mock_paths_config = MagicMock(spec=PathsConfig)
    mock_paths_config.base_dir = pathlib.Path(DUMMY_CONFIG['paths']['base_dir'])
    mock_paths_config.data_dir = pathlib.Path(DUMMY_CONFIG['paths']['data_dir'])
    mock_paths_config.processed_data_file = 'processed_data.pkl'
    mock_paths_config.logs_dir = pathlib.Path('/fake/logs')
    mock_model_config = MagicMock()
    mock_model_config.sequence_length = DUMMY_CONFIG['model']['sequence_length']
    mock_model_config.coordinate_dim = DUMMY_CONFIG['model']['coordinate_dim']
    mock_model_config.n_components = DUMMY_CONFIG['model']['n_components']
    mock_model_config.feature_dim = None # Will be set during extraction

    return mock_paths_config, mock_model_config


@pytest.fixture
def feature_engineer_instance(mock_configs):
    """Fixture to provide a FeatureEngineer instance with mocked dependencies."""
    _, mock_model_config = mock_configs
    fe = FeatureEngineer(config=mock_model_config)
    yield fe


class TestFeatureEngineer:

    def test_init(self, feature_engineer_instance, mock_configs):
        """Test FeatureEngineer initialization."""
        fe = feature_engineer_instance
        _, mock_model_config = mock_configs
        assert fe.config is mock_model_config

    def test__sequence_to_features(self, feature_engineer_instance):
        """Test converting sequence to features."""
        fe = feature_engineer_instance
        # Define expected features for the sequence 'ACD'
        expected_features = np.array([
            list(AMINO_ACID_PROPERTIES['A'].values()),
            list(AMINO_ACID_PROPERTIES['C'].values()),
            list(AMINO_ACID_PROPERTIES['D'].values())
        ], dtype=np.float32)
        features = fe._sequence_to_features(sequence='ACD')
        np.testing.assert_array_equal(features, expected_features)

    @pytest.mark.parametrize("structure_data, expected_targets", [
        # Standard case with chains and residues
        ({'chains': {'A': [{'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 1, 'y': 2, 'z': 3}]},
                           {'residue_number': 2, 'atoms': [{'atom_name': 'CA', 'x': 4, 'y': 5, 'z': 6}]}]}},
        np.array([[1., 2., 3.], [4., 5., 6.]])),
#         # Empty structure data (no chains key)
        ({'chains': {}},[]),  # Empty chains
#         # Missing 'chains' key
        ({'chains': {'A': []}}, []), # Structure data with a chain but no residues
#         # Structure data with a chain and residues, but no CA atoms
        ({'chains': {'A': [{'residue_number': 1, 'atoms': [{'atom_name': 'N', 'x': 1, 'y': 2, 'z': 3}]}]}},[]),
        # Structure data with multiple chains
        ({'chains': {'A': [{'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 1, 'y': 2, 'z': 3}]}],
                     'B': [{'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 4, 'y': 5, 'z': 6}]}]}},

         np.array([[1., 2., 3.], [4., 5., 6.]])),
        #  Structure data with residues out of order to test sorting
        ({'chains': {'A': [{'residue_number': 2, 'atoms': [{'atom_name': 'CA', 'x': 4, 'y': 5, 'z': 6}]},
                           {'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 1, 'y': 2, 'z': 3}]}]}},
 np.array([[1., 2., 3.], [4., 5., 6.]])),
 ({},[]) # Empty dictionary case
    ])
    def test_structure_to_targets(self, feature_engineer_instance, structure_data, expected_targets):
        """Test converting structure data to targets."""
        fe = feature_engineer_instance
        coords, mask = fe._structure_to_targets(structure_data)
        # You can add assertions for the mask here if needed        
        print(structure_data, coords)
        np.testing.assert_array_equal(coords, expected_targets)

    @pytest.mark.parametrize("data, target_length, expected_padded_data", [
        (np.array([[1, 2], [3, 4]]), 4, np.array([[1, 2], [3, 4], [0, 0], [0, 0]])), # Padding
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 2, np.array([[1, 2], [3, 4]])), # Truncation
        (np.array([[1, 2], [3, 4]]), 2, np.array([[1, 2], [3, 4]])), # No change
        (np.array([]).reshape(0, 2), 3, np.array([[0, 0], [0, 0], [0, 0]])), # Padding empty array
        (np.array([[1, 2]]), 0, np.empty((0, 2))) # Truncating to length 0
    ])
    def test_pad_or_truncate(self, feature_engineer_instance, data, target_length, expected_padded_data):
        """Test padding and truncation."""
        fe = feature_engineer_instance # Make sure to get the instance from the fixture
        padded_data = fe._pad_or_truncate(data, target_length, data.shape[-1] if data.ndim > 1 else 0)
        np.testing.assert_array_equal(padded_data, expected_padded_data)

    @patch('src.model.feature_engineering.FeatureEngineer._sequence_to_features')
    @patch('src.model.feature_engineering.FeatureEngineer._structure_to_targets')
    @patch('src.model.feature_engineering.FeatureEngineer._pad_or_truncate')
    def test_extract_features(self, mock_pad_or_truncate, mock_structure_to_targets, mock_sequence_to_features, feature_engineer_instance, mock_configs):
        """Test feature extraction from proteins."""
        fe = feature_engineer_instance
        _, mock_model_config = mock_configs
        dummy_protein_data = {
            'gene_sequence': 'ACD',
            'structure_data': {'chains': {'A': [{'residue_number': 1, 'atoms': [{'atom_name': 'CA', 'x': 1, 'y': 2, 'z': 3}]}]}},
            'pdb_id': '1abc'
        }

        dummy_sequence_features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        dummy_target_coords = np.array([[1, 2, 3]], dtype=np.float32)
        dummy_target_mask = np.array([[1]], dtype=np.float32)

        mock_sequence_to_features.return_value = dummy_sequence_features
        mock_structure_to_targets.return_value = (dummy_target_coords, dummy_target_mask)

        # Configure mock_pad_or_truncate to return padded data
        mock_pad_or_truncate.side_effect = lambda data, max_len, dim: np.zeros((max_len, dim))

        # Call the method under test
        features, targets, gene_sequence = fe.extract_features(dummy_protein_data)

        # Assert that the internal methods were called with the correct arguments
        mock_sequence_to_features.assert_called_once_with(dummy_protein_data.get('gene_sequence'))
        mock_structure_to_targets.assert_called_once_with(dummy_protein_data.get('structure_data'))
        # Add assertions for mock_pad_or_truncate calls if needed

        # Assert that the return values of extract_features are as expected based on the mocks
        # Note: The expected features and targets should match the padded outputs of the mocks
        expected_padded_features = np.zeros((mock_model_config.sequence_length, dummy_sequence_features.shape[-1])) # Based on mock_pad_or_truncate
        expected_padded_targets = np.zeros((mock_model_config.sequence_length, mock_model_config.coordinate_dim)) # Based on mock_pad_or_truncate

        np.testing.assert_array_equal(features, expected_padded_features)
        np.testing.assert_array_equal(targets, expected_padded_targets)

    def test_normalize_features(self, feature_engineer_instance):
        """Test feature normalization."""
        @pytest.mark.parametrize("dummy_features, fit, expected_calls, expected_normalized_features, scaler_mean, scaler_scale", [
            # Standard 2D case, fit=True
            (np.array([[1., 2.], [3., 4.]]), True, {'fit': 1, 'transform': 1}, np.array([[-1., -1.], [1., 1.]]), np.array([2., 3.]), np.array([np.sqrt(2), np.sqrt(2)])),
            # 2D case, fit=False
            (np.array([[5., 6.], [7., 8.]]), False, {'fit': 0, 'transform': 1}, np.array([[1., 1.], [2., 2.]]), np.array([4., 5.]), np.array([1., 1.])),
            # 1D case (should be treated as a single sample with multiple features), fit=True
            (np.array([1., 2., 3.]), True, {'fit': 1, 'transform': 1}, np.array([0., 0., 0.]), np.array([1., 2., 3.]), np.array([1., 1., 1.])), # Assuming single sample normalization centers around itself
            # 3D case (batch of sequences), fit=True
            (np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]), True, {'fit': 1, 'transform': 1}, np.array([[[-1., -1.], [1., 1.]], [[-1., -1.], [1., 1.]]]), np.array([3., 4.]), np.array([np.sqrt(5), np.sqrt(5)])), # Normalized across all samples/timesteps
            # Empty 2D array, fit=True
            (np.empty((0, 2)), True, {'fit': 1, 'transform': 1}, np.empty((0, 2)), np.empty(2), np.empty(2)),
            # Empty 2D array, fit=False
            (np.empty((0, 2)), False, {'fit': 0, 'transform': 1}, np.empty((0, 2)), np.empty(2), np.empty(2)),
        ])
        @patch('sklearn.preprocessing.StandardScaler')
        def test_normalize_features(self, mock_standardscaler, feature_engineer_instance, dummy_features, fit, expected_calls, expected_normalized_features, scaler_mean, scaler_scale):
            """Test feature normalization."""
            
            fe = feature_engineer_instance
            # Configure the return value of transform

            normalized_features = fe.normalize_features(dummy_features, fit=fit)

            if expected_calls['fit'] > 0:
                np.testing.assert_array_equal(mock_scaler_instance.fit.call_args[0][0], dummy_features.reshape(-1, dummy_features.shape[-1]))
            if expected_calls['transform'] > 0:
                 np.testing.assert_array_equal(mock_scaler_instance.transform.call_args[0][0], dummy_features.reshape(-1, dummy_features.shape[-1]))

            np.testing.assert_array_equal(normalized_features, expected_normalized_features)


            