
import logging
from pathlib import Path

class TertiaryParser:
    def __init__(self, pdb_dir: Path):
        self.pdb_dir = pdb_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def extract_atomic_coordinates(self, pdb_file_path: str) -> list[tuple[float, float, float]] | None:
        """
        Extracts atomic coordinates from a PDB file.
        Returns a list of (x, y, z) tuples for each atom.
        """
        if not Path(pdb_file_path).exists():
            logging.error(f"PDB file not found at: {pdb_file_path}")
            return None

        coordinates = []
        try:
            with open(pdb_file_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coordinates.append((x, y, z))
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse coordinates from line: {line}")
        except Exception as e:
            logging.error(f"Error reading or parsing PDB file {pdb_file_path}: {e}")
            return None

        if not coordinates:
            logging.warning(f"No atomic coordinates found in {pdb_file_path}")
            return None

        return coordinates
