"""
PDB Data Downloader and Processor
Downloads PDB structure files from the Protein Data Bank and processes them
to extract structure information using Bio.PDB.
"""

import os
import re
import requests
import gzip
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.PDB import PDBList, PDBParser, MMCIFParser, Selection
from Bio.PDB.PDBExceptions import PDBConstructionException
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
from database_handling.db_models import Atom, ProteinStructure, Residue
from utils.config_loader import load_config
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A simple mapping for three-letter to one-letter amino acid codes
THREE_TO_ONE_LETTER_AA = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
}

class PDBProcessor:
    """
    Handles downloading and parsing of PDB files.
    """
    def __init__(self, data_dir: str = "data/pdb"):
        """
        Initialize PDB processor.

        Args:
            data_dir: Directory to save downloaded PDB files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDBParser(QUIET=True) # Use a single parser instance

    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """
        Download a PDB file from the RCSB PDB website.

        Args:
            pdb_id: The 4-character PDB ID.

        Returns:
            The path to the downloaded file, or None if download fails.
        """
        # The correct download URL uses .pdb.gz instead of .ent.gz
        url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb.gz"
        file_path = self.data_dir / f"{pdb_id.lower()}.pdb.gz"

        # Check if file already exists
        if file_path.exists():
            logger.info(f"File for {pdb_id} already exists. Skipping download.")
            return file_path

        logger.info(f"Downloading PDB ID {pdb_id} from {url}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            with open(file_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded {pdb_id}.")
            return file_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDB ID {pdb_id}: {e}")
            return None

    def parse_pdb_file(self, file_path: Path) -> Optional[ProteinStructure]:
        """
        Parse a PDB file to extract structure information.

        Args:
            file_path: Path to the PDB file.

        Returns:
            A dataclass object representing the protein structure, or None if parsing fails.
        """
        pdb_id = file_path.stem.split('.')[0]
        logger.info(f"Parsing PDB file {file_path}...")

        try:
            # PDB files are often gzipped
            with gzip.open(file_path, 'rt') as f:
                structure = self.parser.get_structure(pdb_id, f)

            # Extract metadata and structure data
            header = structure.header
            title = header.get('name', 'Untitled')
            resolution = header.get('resolution')
            experimental_method = header.get('experimental_method')

            protein_structure = ProteinStructure(
                pdb_id=pdb_id,
                title=title,
                resolution=resolution,
                experimental_method=experimental_method
            )

            # Traverse the structure to get chains, residues, and atoms
            seqres_sequence = ""
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    residues_list: List[Residue] = []

                    for residue in Selection.unfold_entities(chain, 'R'):
                        resname = residue.get_resname()
                        res_num = residue.get_id()[1]

                        # Get a one-letter code, defaulting to 'X' for unknown
                        one_letter_aa = THREE_TO_ONE_LETTER_AA.get(resname, 'X')
                        seqres_sequence += one_letter_aa

                        residue_atoms: List[Atom] = []
                        for atom in Selection.unfold_entities(residue, 'A'):
                            atom_id = atom.get_serial_number()
                            atom_name = atom.get_name()
                            coords = atom.get_coord()
                            occupancy = atom.get_occupancy()
                            b_factor = atom.get_bfactor()
                            element = atom.get_id()[0] # The element symbol is the first char of atom id

                            residue_atoms.append(Atom(
                                atom_id=atom_id,
                                atom_name=atom_name,
                                residue_name=resname,
                                chain_id=chain_id,
                                residue_number=res_num,
                                x=float(coords[0]),
                                y=float(coords[1]),
                                z=float(coords[2]),
                                occupancy=float(occupancy),
                                b_factor=float(b_factor),
                                element=element
                            ))

                        residues_list.append(Residue(
                            residue_number=res_num,
                            residue_name=resname,
                            chain_id=chain_id,
                            atoms=residue_atoms
                        ))

                    protein_structure.chains[chain_id] = residues_list

            protein_structure.seqres_sequence = seqres_sequence

            logger.info(f"Successfully parsed structure for {pdb_id}.")
            return protein_structure

        except PDBConstructionException as e:
            logger.error(f"PDB parsing failed for {pdb_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing {pdb_id}: {e}")
            return None

    def get_chain_sequence(self, protein_structure: ProteinStructure, chain_id: str) -> str:
        """
        Extracts the amino acid sequence for a specific chain from the parsed structure.
        """
        if chain_id not in protein_structure.chains:
            return ""

        sequence = ""
        for residue in protein_structure.chains[chain_id]:
            # Use the three-to-one letter map
            one_letter_aa = THREE_TO_ONE_LETTER_AA.get(residue.residue_name, 'X')
            sequence += one_letter_aa
        return sequence

def main():
    """Example usage of PDBProcessor."""
    try:
        # Load paths config
        config_dir = Path(__file__).parent.parent / 'config'
        paths_config_dict = load_config(config_dir / 'paths.yaml')

        # Convert relative paths to absolute paths
        paths_config = type('PathsConfig', (object,), {
            k: Path(paths_config_dict['paths']['base_dir']) / v if k != 'base_dir' else v
            for k, v in paths_config_dict['paths'].items()
        })

        processor = PDBProcessor(data_dir=paths_config.data_dir / paths_config.pdb_download_dir)

        pdb_ids_to_process = ["1CRN", "4HHB", "2GB1", "1UBQ"]

        downloaded_files: List[Path] = []
        for pdb_id in pdb_ids_to_process:
            file_path = processor.download_pdb(pdb_id)
            if file_path:
                downloaded_files.append(file_path)

        processed_structures: List[ProteinStructure] = []
        for file_path in downloaded_files:
            structure = processor.parse_pdb_file(file_path)
            if structure:
                processed_structures.append(structure)

        logger.info(f"Successfully processed {len(processed_structures)} structures.")

        for structure in processed_structures:
            print(f"\n--- Structure: {structure.pdb_id} ---")
            print(f"Title: {structure.title}")
            print(f"Resolution: {structure.resolution} Å")
            print(f"Experimental Method: {structure.experimental_method}")

            for chain_id, residues in structure.chains.items():
                chain_sequence = processor.get_chain_sequence(structure, chain_id)
                print(f"  Chain {chain_id} sequence: {chain_sequence[:70]}... ({len(chain_sequence)} residues)")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred in the main example: {e}", exc_info=True)

if __name__ == "__main__":
    main()
