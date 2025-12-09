
import os
import logging
from pathlib import Path
from pdb_processor import PDBProcessor
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_handling.database_manager import DatabaseManager
from database_handling.db_models import Base
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PDB_DOWNLOAD_DIR = Path("data/pdb_files")


class PDBDataPipeline:
    def __init__(self, pdb_download_dir: Path, db_manager: DatabaseManager):
        self.pdb_download_dir = pdb_download_dir
        self.pdb_processor = PDBProcessor(str(pdb_download_dir))
        self.db_manager = db_manager

    def process_pdb_ids(self, pdb_ids: list[str]):
        """
        Processes a list of PDB IDs: downloads, parses, and saves to the database.
        """
        with self.db_manager.get_session() as session:
            for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
                try:
                    # Check if the protein already exists in the database
                    if self.db_manager.get_protein_by_pdb_id(pdb_id):
                        logging.info(f"Protein {pdb_id} already exists in the database. Skipping.")
                        continue

                    # Download PDB file
                    pdb_file_path = self.pdb_processor.download_pdb(pdb_id)
                    if not pdb_file_path:
                        logging.warning(f"Skipping {pdb_id} due to download failure.")
                        continue

                    # Parse the PDB file to get the full structure
                    structure_data = self.pdb_processor.parse_pdb(pdb_file_path)
                    if not structure_data or not structure_data.get("chains"):
                        logging.warning(f"Could not parse structure data for {pdb_id}. Skipping.")
                        continue

                    # For simplicity, we can derive the sequence from the parsed structure 
                    # This ensures consistency. Let's take the sequence from the first chain.
                    first_chain = next(iter(structure_data["chains"].values()), None)
                    if not first_chain:
                        logging.warning(f"No chains found in {pdb_id}. Skipping.")
                        continue
                    
                    # Note: This part is simplified. A more robust solution would handle multiple chains.
                    # The feature engineering script now generates the sequence from the structure data directly.
                    gene_sequence = "" # This field is now redundant

                    # Create and save the protein record
                    protein_record = {
                        "pdb_id": pdb_id,
                        "gene_sequence": gene_sequence, # This is no longer the source of truth
                        "structure_data": structure_data # Save the entire parsed structure
                    }
                    new_protein = self.db_manager.create_protein(protein_record)
                    session.add(new_protein)
                    session.commit()

                    logging.info(f"Successfully processed and saved {pdb_id}.")

                except Exception as e:
                    logging.error(f"An error occurred while processing {pdb_id}: {e}", exc_info=True)
                    session.rollback() # Rollback on error

def main():
    # Load database configuration
    config_path = Path(__file__).parent.parent / 'config' / 'database.yaml'
    with open(config_path, 'r') as f:
        db_config = yaml.safe_load(f)
    db_manager = DatabaseManager(db_url=db_config['database']['database_url'])
    db_manager.create_tables(Base) # Ensure tables are created

    pipeline = PDBDataPipeline(PDB_DOWNLOAD_DIR, db_manager)
    
    # Load PDB IDs from the index file
    pdb_ids_to_process = []
    try:
        with open('/home/user/protein-structure/src/data/entries.idx', 'r') as fh:
            # Skipping the header lines
            for line in fh.readlines()[2:]:
                pdb_ids_to_process.append(line.split('\t')[0].lower())
    except FileNotFoundError:
        logging.error("entries.idx not found. Please ensure the data file is in the correct location.")
        return

    if pdb_ids_to_process:
        logging.info(f"Found {len(pdb_ids_to_process)} PDB IDs to process.")
        pipeline.process_pdb_ids(pdb_ids_to_process)
        print(f"\nPipeline finished.")
    else:
        logging.warning("No PDB IDs to process.")

if __name__ == "__main__":
    main()
