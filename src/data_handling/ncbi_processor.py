"""
NCBI Data Downloader and Processor
Downloads and processes protein sequences from NCBI
"""

import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import logging
from urllib.parse import urlencode
import re
from typing import Optional, Any
from database_handling.db_models import ProteinSequence
from utils.config_loader import load_config
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NCBIProteinInfo:
    """Dataclass to hold parsed protein information from NCBI"""
    accession: str
    title: str
    length: int
    organism: str
    gene_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    pdb_id: Optional[str] = None
    sequence: Optional[str] = None

class NCBIProcessor:
    def __init__(self, email: str, tool: str = "protein_structure_predictor"):
        """
        Initialize NCBI processor

        Args:
            email: Your email address (required by NCBI)
            tool: Tool name for NCBI requests
        """
        if not email or email == "your_email@example.com":
            logger.warning("NCBI API requires an email address. Using a placeholder. "
                           "Please update config/credentials.yaml with your email.")
        self.email = email
        self.tool = tool
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.retries = 3
        self.delay = 1.0 # Initial delay for retries

    def _esearch(self, db: str, query: str) -> List[str]:
        """
        Performs an Entrez ESearch to get a list of IDs.

        Args:
            db: The NCBI database to search (e.g., "protein").
            query: The search query.

        Returns:
            A list of matching IDs.
        """
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': db,
            'term': query,
            'usehistory': 'y',
            'retmax': 10000, # Max number of IDs to retrieve
            'tool': self.tool,
            'email': self.email
        }

        for i in range(self.retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                ids = [id_elem.text for id_elem in root.findall('.//Id')]
                return ids
            except requests.exceptions.RequestException as e:
                logger.error(f"ESearch failed (attempt {i+1}/{self.retries}): {e}")
                time.sleep(self.delay * (2**i))
        return []

    def _efetch(self, db: str, ids: List[str], retmode: str = 'xml') -> Optional[Any]:
        """
        Performs an Entrez EFetch to retrieve data for a list of IDs.

        Args:
            db: The NCBI database to fetch from (e.g., "protein").
            ids: A list of IDs.
            retmode: The return mode (e.g., 'xml', 'fasta').

        Returns:
            The raw response content.
        """
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': db,
            'id': ','.join(ids),
            'retmode': retmode,
            'tool': self.tool,
            'email': self.email
        }

        for i in range(self.retries):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                logger.error(f"EFetch failed (attempt {i+1}/{self.retries}): {e}")
                time.sleep(self.delay * (2**i))
        return None

    def search_protein_by_pdb(self, pdb_id: str) -> List[str]:
        """
        Searches the NCBI Protein database for records associated with a PDB ID.

        Args:
            pdb_id: The 4-character PDB ID.

        Returns:
            A list of NCBI accession IDs.
        """
        query = f"PDB:{pdb_id}[All Fields]"
        logger.info(f"Searching NCBI for PDB ID {pdb_id}...")
        return self._esearch(db="protein", query=query)

    def fetch_protein_details(self, accession_ids: List[str]) -> List[NCBIProteinInfo]:
        """
        Fetches detailed information for a list of NCBI accession IDs.

        Args:
            accession_ids: A list of NCBI accession IDs.

        Returns:
            A list of NCBIProteinInfo dataclass objects.
        """
        if not accession_ids:
            return []

        xml_content = self._efetch(db="protein", ids=accession_ids, retmode='xml')
        if not xml_content:
            return []

        proteins = []
        try:
            root = ET.fromstring(xml_content)
            for protein_record in root.findall('.//GBSeq'):
                try:
                    accession = protein_record.find('.//GBSeq_accession-version').text
                    title = protein_record.find('.//GBSeq_definition').text
                    length = int(protein_record.find('.//GBSeq_length').text)
                    organism = protein_record.find('.//GBSeq_organism').text
                    sequence = protein_record.find('.//GBSeq_sequence').text

                    # Extract PDB ID from db_xref
                    pdb_id = None
                    uniprot_id = None
                    gene_name = None

                    db_xrefs = protein_record.findall('.//GBXref_dbname')
                    for xref in db_xrefs:
                        if xref.text == 'PDB':
                            db_id = xref.findall('..//GBXref_id')[0].text
                            # The PDB xref can be in the format '1XYZ' or '1XYZ:A'
                            pdb_id = db_id.split(':')[0]
                        if xref.text == 'UniProtKB/Swiss-Prot':
                            uniprot_id = xref.findall('..//GBXref_id')[0].text

                    # Extract gene name from qualifiers
                    for qualifier in protein_record.findall('.//GBQualifier'):
                        if qualifier.find('GBQualifier_name').text == 'gene':
                            gene_name = qualifier.find('GBQualifier_value').text

                    proteins.append(NCBIProteinInfo(
                        accession=accession,
                        title=title,
                        length=length,
                        organism=organism,
                        gene_name=gene_name,
                        uniprot_id=uniprot_id,
                        pdb_id=pdb_id,
                        sequence=sequence
                    ))
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Failed to parse a protein record from NCBI XML: {e}")
                    continue
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML from NCBI EFetch: {e}")
            return []

        return proteins

    def get_proteins_for_pdb_list(self, pdb_ids: List[str]) -> Dict[str, List[NCBIProteinInfo]]:
        """
        Gets all protein records from NCBI for a list of PDB IDs.

        Args:
            pdb_ids: A list of 4-character PDB IDs.

        Returns:
            A dictionary mapping PDB IDs to a list of matching NCBIProteinInfo objects.
        """
        pdb_to_proteins = {}
        for pdb_id in pdb_ids:
            accession_ids = self.search_protein_by_pdb(pdb_id)
            if accession_ids:
                proteins = self.fetch_protein_details(accession_ids)
                proteins_with_pdb_id = [p for p in proteins if p.pdb_id and p.pdb_id.upper() == pdb_id.upper()]
                pdb_to_proteins[pdb_id] = proteins_with_pdb_id
            else:
                pdb_to_proteins[pdb_id] = []
        return pdb_to_proteins

def main():
    """Example usage of NCBIProcessor."""
    try:
        # Load credentials config
        creds_config_dict = load_config("config/credentials.yaml")['credentials']

        processor = NCBIProcessor(email=creds_config_dict['ncbi_email'])

        # Example PDB IDs
        pdb_ids = ["1A0O", "1AKE", "1BCC", "1CRN", "1UBQ"]

        # Get protein sequences for PDB entries
        pdb_to_proteins = processor.get_proteins_for_pdb_list(pdb_ids)

        # Display results
        for pdb_id, proteins in pdb_to_proteins.items():
            print(f"\nPDB {pdb_id}: {len(proteins)} proteins found")
            for protein in proteins:
                print(f"  {protein.accession}: {protein.title[:60]}...")
                print(f"    Length: {protein.length}, Organism: {protein.organism}")

    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading configuration files: {e}")

if __name__ == "__main__":
    main()
