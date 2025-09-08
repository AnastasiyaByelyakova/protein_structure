"""
Main Data Processing Pipeline for Protein Structure Prediction
Coordinates data acquisition, processing, and storage.
"""

import os
import re
import requests
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from contextlib import contextmanager
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import asdict  # Import asdict for serializing dataclasses
from tqdm import tqdm
from utils.config_loader import load_config
from pdb_processor import PDBProcessor
from ncbi_processor import NCBIProcessor
from database_handling.database_manager import DatabaseManager  # Assuming DatabaseManager is still separate
from database_handling.db_models import Base, Protein, ProteinStructure, ProteinSequence
from pathlib import Path
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use the config_loader to get paths from paths.yaml, similar to app.py
config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
try:
    paths_config = load_config(os.path.join(config_dir, 'paths.yaml'))
    # Use the loaded configuration for the PDB download directory
    PDB_DOWNLOAD_DIR = paths_config['paths']['pdb_download_dir']
except FileNotFoundError:
    logger.error("Could not find paths.yaml. Using default path 'data/pdb'.")
    PDB_DOWNLOAD_DIR = "data/pdb"
try:
    paths_config = load_config(os.path.join(config_dir, 'credentials.yaml'))
    print(paths_config)
    # Use the loaded configuration for the PDB download directory
    NCBI_EMAIL = paths_config['credentials']['ncbi_email']
except FileNotFoundError:
    logger.error("Could not find ncbi_email.yaml. Using default path 'data/pdb'.")
try:
    paths_config = load_config(os.path.join(config_dir, 'database.yaml'))
    print(paths_config)
    # Use the loaded configuration for the PDB download directory
    DATABASE_URL = paths_config['database']['database_url']
except FileNotFoundError:
    logger.error("Could not find ncbi_email.yaml. Using default path 'data/pdb'.")

class MainPipeline:

    def __init__(self, pdb_ids: Optional[List[str]] = None, PDB_DOWNLOAD_DIR = None):
        """
        Initialize the main pipeline with all processors.

        Args:
            pdb_ids: A list of PDB IDs to process. If None, uses a default list.
            config: A configuration dictionary.
        """
        self.pdb_ids = pdb_ids if pdb_ids is not None else EXAMPLE_PDB_IDS
        print(Path.cwd().parent)
        PDB_DOWNLOAD_DIR = os.path.join(Path.cwd().parent, PDB_DOWNLOAD_DIR)
        self.pdb_processor = PDBProcessor(data_dir=PDB_DOWNLOAD_DIR)

        # Load credentials for NCBI
        try:
            creds_config_dict = load_config(os.path.join(config_dir, 'credentials.yaml'))['credentials']
            self.ncbi_processor = NCBIProcessor(email=creds_config_dict['ncbi_email'])
        except FileNotFoundError:
            logger.error("Could not find credentials.yaml. NCBIProcessor will use a placeholder email.")
            self.ncbi_processor = NCBIProcessor(email='your_email@example.com')

        self.db_manager = DatabaseManager(db_url=DATABASE_URL) # DATABASE_URL is from config.py

    def _fetch_all_pdb_ids(self) -> List[str]:
        """
        Fetches a list of all PDB IDs from the PDB archive by parsing pdb_seqres.txt.

        Returns:
            List of PDB IDs
        """
        # The logic to fetch all PDB IDs is moved to the MainPipeline class
        # This implementation requires an update to the class to make it self-contained
        # For now, we'll keep the example PDB IDs
        return self.pdb_ids

    def run(self):
        """
        Execute the main pipeline.
        1. Initialize database (create tables).
        2. Fetch protein data from PDB and NCBI.
        3. Store data in the database.
        """
        logger.info("--- Starting Protein Structure Data Pipeline ---")

        # Step 1: Initialize the database
        logger.info("Initializing database and creating tables...")
        self.db_manager.create_tables(Base)
        logger.info("Database initialized successfully.")

        # Step 2: Fetch and process data for each PDB ID
        pdb_ids_to_process = self._fetch_all_pdb_ids() # Now uses the internal method
        # We'll use a tqdm progress bar for better user feedback
        for pdb_id in tqdm(pdb_ids_to_process, desc="Processing PDB files"):
            try:
                # 2a: Download PDB file
                pdb_file_path = self.pdb_processor.download_pdb(pdb_id)
                if not pdb_file_path:
                    logger.warning(f"Skipping {pdb_id}: PDB file could not be downloaded.")
                    continue

                # 2b: Parse PDB file to get structure data
                protein_structure = self.pdb_processor.parse_pdb_file(pdb_file_path)
                if not protein_structure:
                    logger.warning(f"Skipping {pdb_id}: PDB file could not be parsed.")
                    continue

                # 2c: Get protein sequences and metadata from NCBI
                # Call the correct method, which takes a list of PDB IDs
                ncbi_proteins_dict = self.ncbi_processor.get_proteins_for_pdb_list([pdb_id])

                # Retrieve the list of proteins for the current PDB ID
                ncbi_proteins = ncbi_proteins_dict.get(pdb_id, [])

                # For simplicity, we'll assume the first matching NCBI protein is the one to use
                ncbi_protein = ncbi_proteins[0] if ncbi_proteins else None

                # 2d: Aggregate data and prepare for storage
                # We need to serialize the structure data (dataclass) to a dictionary for JSONB storage
                structure_data_dict = asdict(protein_structure) if protein_structure else None
                entry_metadata_dict = asdict(ncbi_protein) if ncbi_protein else {}

                # 2e: Store or update the protein in the database
                # Add a check to see if the protein already exists
                with self.db_manager.get_session() as session:
                    existing_protein = session.query(Protein).filter_by(pdb_id=pdb_id).first()

                    gene_sequence = protein_structure.seqres_sequence if protein_structure else None

                    if existing_protein:
                        logger.info(f"Updating existing protein record for {pdb_id}.")
                        existing_protein.protein_name = protein_structure.title if protein_structure else None
                        existing_protein.gene_sequence = gene_sequence
                        existing_protein.structure_data = structure_data_dict
                        existing_protein.entry_metadata = entry_metadata_dict
                        existing_protein.ncbi_accession = ncbi_protein.accession if ncbi_protein else None
                        existing_protein.uniprot_id = ncbi_protein.uniprot_id if ncbi_protein else None
                    else:
                        logger.info(f"Creating new protein record for {pdb_id}.")
                        new_protein = Protein(
                            pdb_id=pdb_id,
                            protein_name=protein_structure.title if protein_structure else None,
                            gene_sequence=gene_sequence,
                            structure_data=structure_data_dict,
                            entry_metadata=entry_metadata_dict,
                            ncbi_accession=ncbi_protein.accession if ncbi_protein else None,
                            uniprot_id=ncbi_protein.uniprot_id if ncbi_protein else None
                        )
                        session.add(new_protein)

                    session.commit()

            except Exception as e:
                logger.error(f"Failed to process {pdb_id}: {e}", exc_info=True)

        logger.info("--- Pipeline finished successfully ---")

def main(PDB_DOWNLOAD_DIR):
    """Main function to run the pipeline."""
    pipeline = MainPipeline(PDB_DOWNLOAD_DIR=PDB_DOWNLOAD_DIR)
    pipeline.run()

def get_all_pdb_ids():
    """
    Get list of all current PDB IDs by parsing pdb_seqres.txt.

    Returns:
        List of PDB IDs
    """

    pdb_all_ids_url = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"

    print(f"Fetching list of all PDB IDs from {pdb_all_ids_url}...")
    pdb_ids = set() # Use a set to store unique PDB IDs
    try:
        response = requests.get(pdb_all_ids_url, stream=True, timeout=60)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Read the file line by line to avoid loading large file into memory
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('>'):
                # Example line: >1ABC_1 mol:protein length:100 1ABC_A
                # We want the 4-character PDB ID, which is typically at the start of the line after '>'
                match = re.match(r'^>(\w{4})_', line)
                if match:
                    pdb_ids.add(match.group(1).upper()) # Add unique PDB ID

        all_ids_list = sorted(list(pdb_ids)) # Convert to list and sort
        print(f"Found {len(all_ids_list)} unique PDB IDs from pdb_seqres.txt.")
        return all_ids_list
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download pdb_seqres.txt from {self.pdb_all_ids_url}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing pdb_seqres.txt: {e}")
        return []

# A small list of example PDB IDs to demonstrate the pipeline.
EXAMPLE_PDB_IDS = get_all_pdb_ids()[1467:]

if __name__ == "__main__":
    main(PDB_DOWNLOAD_DIR)
