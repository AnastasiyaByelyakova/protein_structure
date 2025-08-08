import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from numpy.testing import assert_array_equal
import os
import gzip
from pathlib import Path
import requests
from Bio.PDB import PDBParser, Selection
from Bio.PDB.PDBExceptions import PDBConstructionException

# Mock the Bio.PDB components used in PDBProcessor
mock_parser = MagicMock()
mock_structure = MagicMock()
mock_model = MagicMock()
mock_chain = MagicMock()
mock_residue = MagicMock()
mock_atom = MagicMock()

# Configure mocks to allow traversal
mock_parser.get_structure.return_value = mock_structure
mock_structure.__iter__.return_value = [mock_model]
mock_model.__iter__.return_value = [mock_chain]
mock_chain.__iter__.return_value = [mock_residue]
mock_residue.__iter__.return_value = [mock_atom]

# Mock the Selection.unfold_entities function
mock_unfold_entities = MagicMock()
mock_unfold_entities.side_effect = lambda entities, level: entities # Default behavior: return the entity itself

# Dummy data for mocking Bio.PDB objects
dummy_atom_data = {'atom_id': 1, 'atom_name': 'CA', 'residue_name': 'ALA', 'chain_id': 'A', 'residue_number': 10,
                   'x': 1.0, 'y': 2.0, 'z': 3.0, 'occupancy': 1.0, 'b_factor': 10.0, 'element': 'C'}
dummy_residue_data = {'residue_number': 10, 'residue_name': 'ALA', 'chain_id': 'A', 'atoms': [dummy_atom_data]}
dummy_chain_data = {'A': [dummy_residue_data]}
dummy_structure_header = {'name': 'Test Protein', 'resolution': 2.5, 'experimental_method': 'X-RAY DIFFRACTION'}

# Configure mock Bio.PDB objects with dummy data
mock_structure.header = dummy_structure_header
mock_chain.get_id.return_value = 'A'
mock_residue.get_resname.return_value = dummy_residue_data['residue_name']
mock_residue.get_id.return_value = (' ', dummy_residue_data['residue_number'], ' ') # Bio.PDB residue id format
mock_atom.get_serial_number.return_value = dummy_atom_data['atom_id']
mock_atom.get_name.return_value = dummy_atom_data['atom_name']
mock_atom.get_coord.return_value = np.array([dummy_atom_data['x'], dummy_atom_data['y'], dummy_atom_data['z']])
mock_atom.get_occupancy.return_value = dummy_atom_data['occupancy']
mock_atom.get_bfactor.return_value = dummy_atom_data['b_factor']
mock_atom.get_id.return_value = (dummy_atom_data['element'], 0) # Bio.PDB atom id format


# Assume the PDBProcessor class is in src.data_handling.pdb_processor
# Adjust import if necessary
try:
    from src.data_handling.pdb_processor import PDBProcessor, ProteinStructure, Residue, Atom, THREE_TO_ONE_LETTER_AA
except ImportError:
    # Fallback for different test execution environments
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from data_handling.pdb_processor import PDBProcessor, ProteinStructure, Residue, Atom, THREE_TO_ONE_LETTER_AA # type: ignore

class TestPDBProcessor:

    @patch('pathlib.Path.mkdir')
    def test_init(self, mock_mkdir):
        """Test PDBProcessor initialization."""
        processor = PDBProcessor(data_dir="test_data")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert isinstance(processor.data_dir, Path)
        assert processor.data_dir == Path("test_data")

    @patch('src.data_handling.pdb_processor.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_pdb_success(self, mock_file_open, mock_requests_get):
        """Test successful PDB download."""
        processor = PDBProcessor(data_dir="test_data")
        mock_response = MagicMock()
        mock_response.content = b"dummy pdb content"
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        pdb_id = "1abc"
        expected_filepath = Path("test_data") / "1abc.pdb.gz"
        downloaded_filepath = processor.download_pdb(pdb_id)
        mock_requests_get.assert_called_once_with(f"https://files.rcsb.org/download/{pdb_id}.pdb.gz", timeout=30)
        mock_response.raise_for_status.assert_called_once()
        mock_file_open.assert_called_once_with(expected_filepath, 'wb')
        mock_file_open().write.assert_called_once_with(b"dummy pdb content")
        assert downloaded_filepath == expected_filepath

    def test_download_pdb_exists(self):
        """Test PDB download when file already exists."""
        processor = PDBProcessor(data_dir="test_data")
        pdb_id = "1hke"
        expected_filepath = Path("test_data") / "1hke.pdb.gz"
        downloaded_filepath = processor.download_pdb(pdb_id)
        print(downloaded_filepath, expected_filepath)
        assert downloaded_filepath == expected_filepath

    def test_download_pdb_failure(self):
        """Test PDB download failure."""
        processor = PDBProcessor(data_dir="test_data")
        pdb_id = "1abc"
        expected_filepath = Path("test_data") / "1abc.pdb.gz"
        downloaded_filepath = processor.download_pdb(pdb_id)
        assert downloaded_filepath is None

    @patch('src.data_handling.pdb_processor.Selection.unfold_entities', side_effect=mock_unfold_entities)
    @patch('src.data_handling.pdb_processor.gzip.open')
    @patch('src.data_handling.pdb_processor.PDBParser.get_structure', return_value=mock_structure)
    def test_parse_pdb_file_success(self, mock_get_structure, mock_gzip_open, mock_unfold_entities):
        """Test successful PDB file parsing."""
        processor = PDBProcessor(data_dir="test_data")
        dummy_file_path = Path("test_data") / "1abc.pdb.gz"
        dummy_pdb_content = b"dummy pdb content"

        # Configure mock gzip.open to return dummy content
        mock_gzip_open.return_value.__enter__.return_value.read.return_value = dummy_pdb_content.decode('utf-8')

        # Configure mocks to return dummy protein structure data
        mock_model.__iter__.return_value = [mock_chain]
        mock_chain.get_id.return_value = 'A'
        mock_unfold_entities.side_effect = lambda entities, level: [mock_residue] if level == 'R' else [mock_atom] if level == 'A' else []

        mock_residue.get_resname.return_value = 'ALA'
        mock_residue.get_id.return_value = (' ', 10, ' ')
        mock_atom.get_serial_number.return_value = 1
        mock_atom.get_name.return_value = 'CA'
        mock_atom.get_coord.return_value = np.array([1.0, 2.0, 3.0])
        mock_atom.get_occupancy.return_value = 1.0
        mock_atom.get_bfactor.return_value = 10.0
        mock_atom.get_id.return_value = ('C', 0)


        parsed_structure = processor.parse_pdb_file(dummy_file_path)

        mock_gzip_open.assert_called_once_with(dummy_file_path, 'rt')
        mock_get_structure.assert_called_once_with("1abc", mock_gzip_open.return_value.__enter__.return_value)
        mock_unfold_entities.assert_any_call(mock_chain, 'R')
        mock_unfold_entities.assert_any_call(mock_residue, 'A')


        assert isinstance(parsed_structure, ProteinStructure)
        assert parsed_structure.pdb_id == "1abc"
        assert parsed_structure.title == dummy_structure_header['name']
        assert parsed_structure.resolution == dummy_structure_header['resolution']
        assert parsed_structure.experimental_method == dummy_structure_header['experimental_method']
        assert 'A' in parsed_structure.chains
        assert len(parsed_structure.chains['A']) == 1
        assert parsed_structure.chains['A'][0].residue_number == dummy_residue_data['residue_number']
        assert parsed_structure.chains['A'][0].residue_name == dummy_residue_data['residue_name']
        assert parsed_structure.chains['A'][0].chain_id == dummy_residue_data['chain_id']
        assert len(parsed_structure.chains['A'][0].atoms) == 1
        assert parsed_structure.chains['A'][0].atoms[0].atom_name == dummy_atom_data['atom_name']
        assert parsed_structure.seqres_sequence == THREE_TO_ONE_LETTER_AA.get(dummy_residue_data['residue_name'], 'X')


    @patch('src.data_handling.pdb_processor.gzip.open')
    @patch('src.data_handling.pdb_processor.PDBParser.get_structure')
    def test_parse_pdb_file_failure(self, mock_get_structure, mock_gzip_open):
        """Test PDB file parsing failure."""
        processor = PDBProcessor(data_dir="test_data")
        dummy_file_path = Path("test_data") / "1abc.pdb.gz"
        mock_gzip_open.return_value.__enter__.return_value.read.return_value = "corrupted content"
        mock_get_structure.side_effect = PDBConstructionException("Parsing error")

        parsed_structure = processor.parse_pdb_file(dummy_file_path)

        mock_gzip_open.assert_called_once_with(dummy_file_path, 'rt')
        mock_get_structure.assert_called_once_with("1abc", mock_gzip_open.return_value.__enter__.return_value)
        assert parsed_structure is None

    def test_get_chain_sequence_valid(self):
        """Test extracting sequence for a valid chain."""
        processor = PDBProcessor()
        dummy_structure = ProteinStructure(
            pdb_id="1abc",
            title="Test",
            resolution=None,
            experimental_method=None,
            chains={
                'A': [
                    Residue(residue_number=10, residue_name='ALA', chain_id='A', atoms=[]),
                    Residue(residue_number=11, residue_name='CYS', chain_id='A', atoms=[])
                ],
                'B': [
                     Residue(residue_number=12, residue_name='GLY', chain_id='B', atoms=[])
                ]
            },
            seqres_sequence="AC" # This is actually generated during parsing, but we can set it for the dummy
        )
        sequence = processor.get_chain_sequence(dummy_structure, 'A')
        assert sequence == "AC" # Based on ALA and CYS

    def test_get_chain_sequence_invalid(self):
        """Test extracting sequence for an invalid chain."""
        processor = PDBProcessor()
        dummy_structure = ProteinStructure(
            pdb_id="1abc",
            title="Test",
            resolution=None,
            experimental_method=None,
            chains={
                'A': [
                    Residue(residue_number=10, residue_name='ALA', chain_id='A', atoms=[])
                ]
            },
             seqres_sequence="A"
        )
        sequence = processor.get_chain_sequence(dummy_structure, 'B')
        assert sequence == ""

    def test_get_chain_sequence_unknown_aa(self):
        """Test extracting sequence with unknown amino acids."""
        processor = PDBProcessor()
        dummy_structure = ProteinStructure(
            pdb_id="1abc",
            title="Test",
            resolution=None,
            experimental_method=None,
            chains={
                'A': [
                    Residue(residue_number=10, residue_name='XYZ', chain_id='A', atoms=[]), # Unknown AA
                    Residue(residue_number=11, residue_name='GLY', chain_id='A', atoms=[])
                ]
            },
             seqres_sequence="XG" # Expected sequence with 'X' for unknown
        )
        sequence = processor.get_chain_sequence(dummy_structure, 'A')
        assert sequence == "XG" # Based on XYZ and GLY (XYZ maps to X)