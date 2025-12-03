
import logging
from pathlib import Path

class PrimaryParser:
    def __init__(self, pdb_dir: Path):
        self.pdb_dir = pdb_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_sequence_from_pdb(self, pdb_file_path: str) -> str | None:
        """
        Extracts the amino acid sequence from a PDB file.
        Returns the sequence as a string.
        """
        if not Path(pdb_file_path).exists():
            logging.error(f"PDB file not found at: {pdb_file_path}")
            return None

        sequence = ""
        # A set to keep track of residue numbers we've already added to the sequence
        processed_residues = set()

        try:
            with open(pdb_file_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") and line[13:15].strip() == "CA":
                        residue_name = line[17:20].strip()
                        residue_number = int(line[22:26].strip())
                        chain_id = line[21]
                        
                        # Create a unique identifier for the residue to handle multi-chain proteins
                        residue_id = f"{chain_id}-{residue_number}"
                        
                        if residue_id not in processed_residues:
                            sequence += self._three_to_one_letter_code(residue_name)
                            processed_residues.add(residue_id)

        except Exception as e:
            logging.error(f"Error reading or parsing PDB file {pdb_file_path}: {e}")
            return None

        if not sequence:
            logging.warning(f"No sequence information found in {pdb_file_path}")
            return None

        return sequence

    def _three_to_one_letter_code(self, three_letter_code: str) -> str:
        """Converts a three-letter amino acid code to a one-letter code."""
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return mapping.get(three_letter_code.upper(), '?')  # Return '?' for unknown codes
