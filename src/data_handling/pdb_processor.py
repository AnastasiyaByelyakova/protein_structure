
import os
import urllib.request
from pathlib import Path
import numpy as np

RESIDUE_ATOMS = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND2', 'OD1'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']
}

class PDBProcessor:
    def __init__(self, pdb_download_dir):
        self.pdb_download_dir = Path(pdb_download_dir)
        self.pdb_download_dir.mkdir(parents=True, exist_ok=True)

    def download_pdb(self, pdb_id: str) -> str | None:
        pdb_id = pdb_id.lower()
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_file_path = self.pdb_download_dir / f"{pdb_id}.pdb"

        if pdb_file_path.exists():
            print(f"{pdb_id}.pdb already exists. Skipping download.")
            return str(pdb_file_path)

        try:
            with urllib.request.urlopen(pdb_url) as response:
                if response.status != 200:
                    print(f"Error downloading {pdb_id}: HTTP status {response.status}")
                    return None
                content = response.read().decode('utf-8')
            with open(pdb_file_path, 'w') as f:
                f.write(content)
            return str(pdb_file_path)
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            return None

def get_full_atom_coords(ca_coord, residue_name):
    """
    Generates full atom coordinates for a given residue.
    This is a simplified model and might not be accurate.
    """
    if residue_name not in RESIDUE_ATOMS:
        residue_name = 'ALA'  # Default to ALA

    atoms = RESIDUE_ATOMS[residue_name]
    coords = {}
    
    # Place CA at the given coordinate
    coords['CA'] = np.array(ca_coord)
    
    # Simplified placement of other atoms relative to CA
    if 'N' in atoms:
        coords['N'] = coords['CA'] + np.array([-0.5, 0.8, -0.2])
    if 'C' in atoms:
        coords['C'] = coords['CA'] + np.array([0.6, -0.7, 0.4])
    if 'O' in atoms:
        coords['O'] = coords['C'] + np.array([0.0, -1.2, 0.0])
    if 'CB' in atoms:
        coords['CB'] = coords['CA'] + np.array([-1.2, -0.2, 0.5])
    
    # Add other atoms for larger residues (highly simplified)
    for atom in atoms:
        if atom not in coords:
            coords[atom] = coords['CA'] + np.random.uniform(-1, 1, 3)
            
    return {atom: coords.get(atom, coords['CA']) for atom in atoms}

def coordinates_to_pdb(coordinates: list, sequence: str, chain_id="A") -> str:
    """
    Converts a list of 3D coordinates into a PDB format string for a full atom model.
    """
    pdb_lines = []
    atom_index = 1
    for i, ca_coord in enumerate(coordinates):
        residue_name = sequence[i:i+3].upper()
        if residue_name not in RESIDUE_ATOMS:
            residue_name = "ALA" # Default to ALA if not found
        
        residue_index = i + 1
        
        atom_coords = get_full_atom_coords(ca_coord, residue_name)
        
        for atom_name, atom_coord in atom_coords.items():
            x, y, z = atom_coord
            # PDB format is very specific about spacing.
            line = (
                f"ATOM  {atom_index:>5}  {atom_name:<4} {residue_name} {chain_id}{residue_index:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                '  1.00 20.00           C  '
            )
            pdb_lines.append(line)
            atom_index += 1
        
    pdb_lines.append("TER")
    pdb_lines.append("END")
    
    return "\n".join(pdb_lines)
