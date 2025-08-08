import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from typing import Union
import yaml
from contextlib import contextmanager

# --- SQLAlchemy Database Layer ---
Base = declarative_base()

class Protein(Base):
    """
    SQLAlchemy model for the 'proteins' table in the PostgreSQL database.
    Stores aggregated protein information.
    """
    __tablename__ = 'proteins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pdb_id = Column(String(4), unique=True, nullable=False, index=True)
    protein_name = Column(String(255))
    gene_sequence = Column(Text)  # Full SEQRES sequence for the entry
    structure_data = Column(JSONB)  # Store parsed structure data (chains, residues, atoms)
    entry_metadata = Column(JSONB)  # Renamed from 'metadata' to avoid SQLAlchemy reserved keyword
    ncbi_accession = Column(String(20), index=True)  # From NCBI
    uniprot_id = Column(String(20), index=True)  # From NCBI
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<Protein(pdb_id='{self.pdb_id}', name='{self.protein_name}')>"

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

@dataclass
class Atom:
    """Represents an atom in a protein structure"""
    atom_id: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str

@dataclass
class Residue:
    """Represents a residue in a protein structure"""
    residue_number: int
    residue_name: str
    chain_id: str
    atoms: List[Atom] = field(default_factory=list)

@dataclass
class ProteinStructure:
    """Represents a complete protein structure"""
    pdb_id: str
    title: Optional[str] = None
    resolution: Optional[float] = None
    experimental_method: Optional[str] = None
    chains: Dict[str, List[Residue]] = field(default_factory=dict)
    seqres_sequence: Optional[str] = None

@dataclass
class ProteinSequence:
    """Represents protein sequence information from NCBI"""
    accession: str
    title: str
    length: int
    organism: str
    gene_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    pdb_id: Optional[str] = None
    sequence: Optional[str] = None
