
import os
import urllib.request
from pathlib import Path
import numpy as np
from collections import defaultdict


class PDBProcessor:
    def __init__(self, pdb_download_dir):
        self.pdb_download_dir = Path(pdb_download_dir)
        self.pdb_download_dir.mkdir(parents=True, exist_ok=True)

    def download_pdb(self, pdb_id: str) -> str | None:
        pdb_id = pdb_id.lower()
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_file_path = self.pdb_download_dir / f"{pdb_id}.pdb"

        if pdb_file_path.exists():
            # print(f"{pdb_id}.pdb already exists. Skipping download.")
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

    def parse_pdb(self, file_path: str) -> dict:
        """
        Parses a PDB file and extracts structured data.

        Args:
            file_path: The path to the PDB file.

        Returns:
            A dictionary containing the parsed structure data, organized by chains, 
            residues, and atoms.
        """
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return {}

        structure = {"chains": defaultdict(list)}
        # Using defaultdict to hold residues for a chain
        # Key: residue number, Value: residue dict
        residues_by_chain = defaultdict(dict)

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        atom_info = {
                            "atom_name": line[12:16].strip(),
                            "residue_name": line[17:20].strip(),
                            "chain_id": line[21].strip(),
                            "residue_number": int(line[22:26]),
                            "x": float(line[30:38]),
                            "y": float(line[38:46]),
                            "z": float(line[46:54]),
                        }

                        chain_id = atom_info["chain_id"]
                        res_num = atom_info["residue_number"]
                        res_name = atom_info["residue_name"]

                        # Get or create the residue dictionary
                        residue = residues_by_chain[chain_id].setdefault(res_num, {
                            "residue_name": res_name,
                            "residue_number": res_num,
                            "atoms": []
                        })
                        
                        # Add atom to the residue's atom list
                        residue["atoms"].append(atom_info)

                    except (ValueError, IndexError) as e:
                        # print(f"Skipping malformed ATOM line: {line.strip()} | Error: {e}")
                        continue
        
        # Convert the defaultdict of residues into a sorted list for each chain
        for chain_id, residues in residues_by_chain.items():
            sorted_residues = sorted(residues.values(), key=lambda r: r['residue_number'])
            structure["chains"][chain_id] = sorted_residues

        return structure
