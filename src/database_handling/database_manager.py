"""
Database Manager
Handles PostgreSQL database operations for protein structure prediction project
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text # Import 'text' for raw SQL execution
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base # Import this to ensure Base is available for type hinting
from .db_models import *
# from data_handling.pdb_processor import PDBProcessor # Needed for three_to_one_letter_aa mapping in get_training_dataset
from sqlalchemy import or_ # For OR conditions in search queries
from dataclasses import is_dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize database manager

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            db_url: Optional SQLAlchemy URL. If provided, overrides other connection details.
        """
        # If a full URL is provided, use it. Otherwise, construct one from components.
        if db_url:
            self.db_url = db_url
        else:
            raise ValueError("db_url must be provided for DatabaseManager initialization.")

        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info("Database engine created")

    @contextmanager
    def get_session(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self, base_class: Any):
        """Create all tables defined in the Base class."""
        logger.info("Creating database tables...")
        base_class.metadata.create_all(bind=self.engine)
        logger.info("Tables created successfully.")

    def create_protein(self, protein_data: dict) -> Protein:
        """Creates a new Protein object from a dictionary of data."""
        return Protein(**protein_data)

    def create_indexes(self):
        """Manually create indexes if needed (e.g., for JSONB keys)."""
        # This is more complex in SQLAlchemy and often handled automatically
        # or with explicit `Index` definitions in db_models.py.
        # This function is kept as a placeholder but is not strictly necessary
        # if using the declarative approach with `index=True` on columns.
        pass

    def get_all_proteins(self) -> List[Protein]:
        """
        Retrieves all protein records from the database.
        This is a new method added to resolve the AttributeError.
        """
        with self.get_session() as session:
            proteins = session.query(Protein).all()
            proteins =  [i.to_dict() for i in proteins]
            return proteins

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get statistics about the database, like record counts for each table.
        """
        stats = {}
        with self.get_session() as session:
            # We use text() for raw SQL to get a count from each table.
            stats['proteins'] = session.query(Protein).count()
            # Add other tables here as needed
        return stats

    def get_training_dataset(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves a sample of protein data suitable for model training.

        Args:
            limit: Maximum number of proteins to retrieve.
        Returns:
            A list of dictionaries, where each dictionary contains
            the protein sequence and structure data.
        """
        training_data = []
        with self.get_session() as session:
            proteins = session.query(Protein).limit(limit).all()
            for p in proteins:
                # Ensure structure_data is available and has necessary info
                if p.structure_data and 'chains' in p.structure_data and p.gene_sequence:
                    # Parse the structure data from the stored JSON
                    try:
                        structure = ProteinStructure(**p.structure_data)
                        # Extract the sequence from the structure, as the gene_sequence
                        # might be the full entry, not the one from the PDB file.
                        # This ensures the sequence and structure are aligned.
                        chain_sequences = {
                            chain_id: "".join(
                                PDBProcessor.three_to_one_letter_aa.get(res.residue_name, 'X')
                                for res in residues
                            )
                            for chain_id, residues in structure.chains.items()
                        }
                        # We will train on the concatenated sequence and coordinates
                        full_sequence = "".join(chain_sequences.values())
                        # If the extracted sequence doesn't match the stored one,
                        # we might have a data quality issue. For now, we use the
                        # extracted sequence.
                        if not full_sequence:
                            logger.warning(f"Could not extract a valid sequence from structure for PDB ID: {p.pdb_id}. Skipping.")
                            continue

                        # Extract all atom coordinates
                        all_coordinates = []
                        for chain_id, residues in structure.chains.items():
                            for residue in residues:
                                for atom in residue.atoms:
                                    all_coordinates.append([atom.x, atom.y, atom.z])

                        if not all_coordinates:
                             logger.warning(f"No coordinates found for PDB ID: {p.pdb_id}. Skipping.")
                             continue

                        training_data.append({
                            'pdb_id': p.pdb_id,
                            'sequence': full_sequence,
                            'coordinates': np.array(all_coordinates, dtype=np.float32)
                        })
                    except Exception as e:
                        logger.error(f"Error processing protein {p.pdb_id}: {e}")
                        continue
        return training_data

    def search_proteins(self, query: str, limit: int = 10) -> List[Protein]:
        """
        Searches for proteins by name, PDB ID, or gene sequence.

        Args:
            query: The search string.
            limit: Maximum number of results to return.
        Returns:
            A list of matching Protein objects.
        """
        with self.get_session() as session:
            # Use 'ilike' for case-insensitive partial string matching.
            search_query = (
                session.query(Protein)
                .filter(
                    or_(
                        Protein.protein_name.ilike(f'%{query}%'),
                        Protein.gene_sequence.ilike(f'%{query}%'),
                        Protein.pdb_id.ilike(f'%{query}%'),
                    )
                )
                .limit(limit)
                .all()
            )
        return search_query

if __name__ == "__main__":
    # Example usage for standalone script
    db_manager = DatabaseManager(
        db_url="postgresql://neondb_owner:npg_v7N8UpXlYFun@ep-sweet-band-adaqfu6r-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )

    # Note: create_tables is typically called by main_pipeline.py
    # For standalone testing, you'd need to import Base from db_models.
    stats = db_manager.create_tables(Base)

    # Get database statistics
    stats = db_manager.get_database_stats()
    print("Database Statistics:")
    for table, count in stats.items():
        print(f"  {table}: {count} records")

    # Example: Retrieve a training dataset sample
    training_samples = db_manager.get_training_dataset(limit=2)
    if training_samples:
        print("\nFirst training sample:")
        # Use default=str for numpy arrays, which are not directly JSON serializable
        print(json.dumps(training_samples[0], indent=2, default=str))
    else:
        print("\nNo training samples found. Run main_pipeline.py first.")

    # Example: Search proteins
    print("\nSearching for proteins with 'insulin':")
    search_results = db_manager.search_proteins(query="insulin", limit=5)
    for p in search_results:
        print(f"  - {p.pdb_id}: {p.protein_name}")
