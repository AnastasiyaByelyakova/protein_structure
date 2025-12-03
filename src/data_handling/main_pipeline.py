
import os
import logging
import requests
from pathlib import Path
from pdb_processor import PDBProcessor
from tertiary_parser import TertiaryParser
from primary_parser import PrimaryParser


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PDB_DOWNLOAD_DIR = Path("data/pdb_files")


class PDBDataPipeline:
    def __init__(self, pdb_download_dir: Path):
        self.pdb_download_dir = pdb_download_dir
        self.pdb_processor = PDBProcessor(str(pdb_download_dir))
        self.primary_parser = PrimaryParser(pdb_download_dir)
        self.tertiary_parser = TertiaryParser(pdb_download_dir)
        self.pdb_all_ids_url = "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"

    def process_pdb_ids(self, pdb_ids: list[str]) -> list[dict]:
        """
        Processes a list of PDB IDs: downloads, extracts primary, and tertiary structures.
        """
        processed_data = []
        for pdb_id in pdb_ids:
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

                processed_data.append({
                    'pdb_id': pdb_id,
                    'primary': primary_structure,
                    'tertiary': tertiary_structure
                })
                logging.info(f"Successfully processed {pdb_id}.")

            except Exception as e:
                logging.error(f"An error occurred while processing {pdb_id}: {e}")

        return processed_data

    def get_all_pdb_ids(self):
        """
        Downloads the list of all PDB IDs from the PDB FTP server.
        This is a large file, so it might take some time.
        """
        try:
            response = requests.get(self.pdb_all_ids_url, timeout=60)
            response.raise_for_status()
            # The file contains a list of PDB IDs, one per line.
            # We can split the text by newlines and filter for valid PDB IDs.
            all_ids_list = [line.split()[0].lower() for line in response.text.split('\n') if line and len(line.split()[0]) == 4]
            logging.info(f"Found {len(all_ids_list)} unique PDB IDs from pdb_seqres.txt.")
            print(f"Found {len(all_ids_list)} unique PDB IDs from pdb_seqres.txt.")
            return all_ids_list
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download pdb_seqres.txt from {self.pdb_all_ids_url}: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred while parsing pdb_seqres.txt: {e}")
            return []

# A small list of example PDB IDs to demonstrate the pipeline.
EXAMPLE_PDB_IDS = ['1crn', '2hds', '1abc']

if __name__ == "__main__":
    pipeline = PDBDataPipeline(PDB_DOWNLOAD_DIR)
    # To process all PDB IDs, uncomment the following line:
    # pdb_ids_to_process = pipeline.get_all_pdb_ids()
    
    # For demonstration, we'll use the example list.
    pdb_ids_to_process = EXAMPLE_PDB_IDS

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
