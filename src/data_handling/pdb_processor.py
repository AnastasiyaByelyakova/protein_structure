
def coordinates_to_pdb(coordinates: list, residue_name="ALA", chain_id="A") -> str:
    """
    Converts a list of 3D coordinates into a PDB format string for a C-alpha trace.
    """
    pdb_lines = []
    for i, (x, y, z) in enumerate(coordinates):
        atom_index = i + 1
        residue_index = i + 1
        
        # PDB format is very specific about spacing.
        line = (
            f"ATOM  {atom_index:>5}  CA  {residue_name} {chain_id}{residue_index:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            "  1.00  0.00           C  "
        )
        pdb_lines.append(line)
        
    pdb_lines.append("TER")
    pdb_lines.append("END")
    
    return "\n".join(pdb_lines)
