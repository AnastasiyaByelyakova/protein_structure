
import os
import logging
from pathlib import Path
from pdb_processor import PDBProcessor
from tertiary_parser import TertiaryParser
from primary_parser import PrimaryParser
import sys
sys.path.append('/home/user/protein-structure/src')
from database_handling.database_manager import DatabaseManager
from database_handling.db_models import Base
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PDB_DOWNLOAD_DIR = Path("data/pdb_files")


class PDBDataPipeline:
    def __init__(self, pdb_download_dir: Path, db_manager: DatabaseManager):
        self.pdb_download_dir = pdb_download_dir
        self.pdb_processor = PDBProcessor(str(pdb_download_dir))
        self.primary_parser = PrimaryParser(pdb_download_dir)
        self.tertiary_parser = TertiaryParser(pdb_download_dir)
        self.db_manager = db_manager

    def process_pdb_ids(self, pdb_ids: list[str]) -> list[dict]:
        """
        Processes a list of PDB IDs: downloads, extracts primary, and tertiary structures.
        """
        processed_data = []
        for pdb_id in tqdm(pdb_ids):
            try:
                # Download PDB file
                pdb_file_path = self.pdb_processor.download_pdb(pdb_id)
                if not pdb_file_path:
                    logging.warning(f"Skipping {pdb_id} due to download failure.")
                    continue

                # Extract primary structure (sequence)
                primary_structure = self.primary_parser.get_sequence_from_pdb(pdb_file_path)
                if not primary_structure:
                    logging.warning(f"Could not extract primary structure for {pdb_id}. Skipping.")
                    continue

                # Extract tertiary structure (coordinates)
                tertiary_structure = self.tertiary_parser.extract_atomic_coordinates(pdb_file_path)
                if not tertiary_structure:
                    logging.warning(f"Could not extract tertiary structure for {pdb_id}. Skipping.")
                    continue

                # Save to database
                with self.db_manager.get_session() as session:
                    protein_record = {
                        "pdb_id": pdb_id,
                        "gene_sequence": primary_structure,
                        "structure_data": {"coordinates": tertiary_structure}
                    }
                    # A simple way to insert data, consider a more robust method for production
                    new_protein = self.db_manager.create_protein(protein_record)
                    session.add(new_protein)
                    session.commit()

                processed_data.append({
                    'pdb_id': pdb_id,
                    'primary': primary_structure,
                    'tertiary': tertiary_structure
                })
                logging.info(f"Successfully processed and saved {pdb_id}.")

            except Exception as e:
                logging.error(f"An error occurred while processing {pdb_id}: {e}")

        return processed_data

# A large, diverse list of PDB IDs for your training set
LARGE_PDB_ID_LIST = [
    '1crn', '2hds', '1abc', '1gfl', '2gfl', '3gfl', '4gfl', '5gfl', '6gfl', '1aon',
    '1b2a', '1b3a', '1b4a', '1b5a', '1b6a', '1b7a', '1b8a', '1b9a', '1baa', '1bab',
    '1bac', '1bad', '1bae', '1baf', '1bag', '1bah', '1bai', '1baj', '1bak', '1bal',
    '1bam', '1ban', '1bao', '1bap', '1baq', '1bar', '1bas', '1bat', '1bau', '1bav',
    '1baw', '1bax', '1bay', '1baz', '1bba', '1bbb', '1bbc', '1bbd', '1bbe', '1bbf',
    '1bbg', '1bbh', '1bbi', '1bbj', '1bbk', '1bbl', '1bbm', '1bbn', '1bbo', '1bbp',
    '1bbq', '1bbr', '1bbs', '1bbt', '1bbu', '1bbv', '1bbw', '1bbx', '1bby', '1bbz',
    '1bca', '1bcb', '1bcc', '1bcd', '1bce', '1bcf', '1bcg', '1bch', '1bci', '1bcj',
    '1bck', '1bcl', '1bcm', '1bcn', '1bco', '1bcp', '1bcq', '1bcr', '1bcs', '1bct',
    '1bcu', '1bcv', '1bcw', '1bcx', '1bcy', '1bcz', '1bda', '1bdb', '1bdc', '1bdd'
]

def main():
    # Initialize DatabaseManager
    db_manager = DatabaseManager(db_url="postgresql://neondb_owner:npg_ZyXY5Hfur4TC@ep-icy-math-ahazgfq9-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    db_manager.create_tables(Base)

    pipeline = PDBDataPipeline(PDB_DOWNLOAD_DIR, db_manager)
    
    # We will use the large, hardcoded list of PDB IDs as a workaround
    # pdb_ids_to_process = LARGE_PDB_ID_LIST
    pdb_ids_to_process = []
    with open('/home/user/protein-structure/src/data/entries.idx','r') as fh:
        for i in fh.readlines():
            pdb_ids_to_process.append(i.split('\t')[0])


    if pdb_ids_to_process:
        processed_results = pipeline.process_pdb_ids(pdb_ids_to_process)
        print(f"\nProcessed {len(processed_results)} PDB files.")
        # You can now do something with the processed_results list,
        # like saving it to a file or database.
        # For example, print the first result:
        if processed_results:
            print("\nExample of processed data:")
            print(processed_results[0])
    else:
        logging.warning("No PDB IDs to process.")

if __name__ == "__main__":
    main()
