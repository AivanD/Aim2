import sqlite3
import gzip
import tarfile
import logging
from pathlib import Path
import csv
from tqdm import tqdm
import sys

from aim2.utils.config import DATA_DIR, DATABASE_FILE

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    logger.info(f"Connecting to database at {DATABASE_FILE}...")
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        # --- Optimizations for bulk insert speed ---
        conn.execute('PRAGMA journal_mode = WAL;')       # Use Write-Ahead Logging
        conn.execute('PRAGMA synchronous = NORMAL;')     # Safer setting that waits for OS to receive data
        conn.execute('PRAGMA cache_size = -2000000;')   # Use up to 2GB of RAM for cache
        conn.execute('PRAGMA temp_store = MEMORY;')     # Store temporary tables in RAM
        conn.execute('PRAGMA locking_mode = EXCLUSIVE;')# Exclusive access for this connection
        conn.execute('PRAGMA foreign_keys = OFF;')      # Disable foreign key checks during load
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def create_tables(conn: sqlite3.Connection):
    """Creates the necessary tables in the database if they don't exist."""
    logger.info("Creating database tables...")
    cursor = conn.cursor()
    
    # PubChem Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pubchem_smiles (
        cid INTEGER PRIMARY KEY,
        smiles TEXT NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pubchem_inchikey (
        cid INTEGER PRIMARY KEY,
        inchikey TEXT NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pubchem_synonyms (
        cid INTEGER,
        synonym TEXT NOT NULL,
        FOREIGN KEY (cid) REFERENCES pubchem_smiles(cid)
    );
    """)

    # NCBI Taxonomy Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS taxonomy_nodes (
        tax_id INTEGER PRIMARY KEY,
        parent_tax_id INTEGER,
        rank TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS taxonomy_names (
        tax_id INTEGER,
        name TEXT NOT NULL,
        name_class TEXT,
        FOREIGN KEY (tax_id) REFERENCES taxonomy_nodes(tax_id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS taxonomy_merged (
        old_tax_id INTEGER PRIMARY KEY,
        new_tax_id INTEGER
    );
    """)
    
    conn.commit()
    logger.info("Tables created successfully.")

def create_indexes(conn: sqlite3.Connection):
    """Creates indexes on frequently searched columns for performance."""
    logger.info("Creating database indexes...")
    cursor = conn.cursor()
    
    # Index for fast lookup of synonyms. Must match the query's collation.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonym_nocase ON pubchem_synonyms (synonym COLLATE NOCASE);")
    
    # Index for fast lookup of taxonomy names. Must match the query's collation.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tax_name_nocase ON taxonomy_names (name COLLATE NOCASE);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tax_name_class ON taxonomy_names (name_class);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tax_id_names ON taxonomy_names (tax_id);")

    # Re-enable foreign keys after all data is loaded and indexed
    conn.execute('PRAGMA foreign_keys = ON;')
    conn.commit()
    logger.info("Indexes created successfully.")


def _insert_batch(conn, table, data):
    if not data:
        return
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] * len(data[0]))
    query = f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})"
    cursor.executemany(query, data)
    # The commit will be handled by the calling function

def process_cid_smiles(conn: sqlite3.Connection):
    """Parses CID-SMILES.gz and populates the pubchem_smiles table."""
    file_path = DATA_DIR / "CID-SMILES.gz"
    if not file_path.exists():
        logger.warning(f"{file_path.name} not found, skipping.")
        return
    
    logger.info(f"Processing {file_path.name}...")
    batch_size = 500000
    batch = []
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION;")
    try:
        with gzip.open(file_path, 'rt') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for row in tqdm(reader, desc="Populating pubchem_smiles"):
                batch.append((int(row[0]), row[1]))
                if len(batch) >= batch_size:
                    _insert_batch(conn, 'pubchem_smiles', batch)
                    batch = []
            if batch:
                _insert_batch(conn, 'pubchem_smiles', batch)
        cursor.execute("COMMIT;")
    except Exception as e:
        logger.error(f"Transaction failed for pubchem_smiles: {e}. Rolling back.")
        cursor.execute("ROLLBACK;")
        raise
    logger.info(f"Finished processing {file_path.name}.")

def process_cid_inchikey(conn: sqlite3.Connection):
    """Parses CID-InChI-Key.gz and populates the pubchem_inchikey table."""
    file_path = DATA_DIR / "CID-InChI-Key.gz"
    if not file_path.exists():
        logger.warning(f"{file_path.name} not found, skipping.")
        return
    
    logger.info(f"Processing {file_path.name}...")
    batch_size = 500000
    batch = []
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION;")
    try:
        with gzip.open(file_path, 'rt') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for row in tqdm(reader, desc="Populating pubchem_inchikey"):
                # row is [CID, InChI, InChIKey]
                batch.append((int(row[0]), row[2]))
                if len(batch) >= batch_size:
                    _insert_batch(conn, 'pubchem_inchikey', batch)
                    batch = []
            if batch:
                _insert_batch(conn, 'pubchem_inchikey', batch)
        cursor.execute("COMMIT;")
    except Exception as e:
        logger.error(f"Transaction failed for pubchem_inchikey: {e}. Rolling back.")
        cursor.execute("ROLLBACK;")
        raise
    logger.info(f"Finished processing {file_path.name}.")

def process_cid_synonyms(conn: sqlite3.Connection):
    """Parses CID-Synonym-filtered.gz and populates the pubchem_synonyms table."""
    file_path = DATA_DIR / "CID-Synonym-filtered.gz"
    if not file_path.exists():
        logger.warning(f"{file_path.name} not found, skipping.")
        return
    
    logger.info(f"Processing {file_path.name}...")
    
    # Increase the field size limit for the CSV reader
    # This is necessary because some synonyms can be very long.
    # We'll find the maximum possible limit for the system.
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    batch_size = 500000
    batch = []
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION;")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for row in tqdm(reader, desc="Populating pubchem_synonyms"):
                batch.append((int(row[0]), row[1]))
                if len(batch) >= batch_size:
                    _insert_batch(conn, 'pubchem_synonyms', batch)
                    batch = []
            if batch:
                _insert_batch(conn, 'pubchem_synonyms', batch)
        cursor.execute("COMMIT;")
    except Exception as e:
        logger.error(f"Transaction failed for pubchem_synonyms: {e}. Rolling back.")
        cursor.execute("ROLLBACK;")
        raise
    logger.info(f"Finished processing {file_path.name}.")


def process_taxdump(conn: sqlite3.Connection):
    """Extracts and processes taxdump.tar.gz."""
    file_path = DATA_DIR / "taxdump.tar.gz"
    if not file_path.exists():
        logger.warning(f"{file_path.name} not found, skipping.")
        return

    logger.info(f"Processing {file_path.name}...")
    batch_size = 500000
    cursor = conn.cursor()
    
    with tarfile.open(file_path, "r:gz") as tar:
        # Process nodes.dmp
        logger.info("Processing nodes.dmp...")
        nodes_file = tar.extractfile("nodes.dmp")
        if nodes_file:
            cursor.execute("BEGIN TRANSACTION;")
            try:
                batch = []
                for line in tqdm(nodes_file, desc="Populating taxonomy_nodes"):
                    fields = line.decode('utf-8').split('\t|\t')
                    tax_id = int(fields[0])
                    parent_tax_id = int(fields[1])
                    rank = fields[2]
                    batch.append((tax_id, parent_tax_id, rank))
                    if len(batch) >= batch_size:
                        _insert_batch(conn, 'taxonomy_nodes', batch)
                        batch = []
                if batch:
                    _insert_batch(conn, 'taxonomy_nodes', batch)
                cursor.execute("COMMIT;")
            except Exception as e:
                logger.error(f"Transaction failed for taxonomy_nodes: {e}. Rolling back.")
                cursor.execute("ROLLBACK;")
                raise
        
        # Process names.dmp
        logger.info("Processing names.dmp...")
        names_file = tar.extractfile("names.dmp")
        if names_file:
            cursor.execute("BEGIN TRANSACTION;")
            try:
                batch = []
                for line in tqdm(names_file, desc="Populating taxonomy_names"):
                    fields = line.decode('utf-8').split('\t|\t')
                    tax_id = int(fields[0])
                    name_txt = fields[1]
                    name_class = fields[3].strip().replace('\t|', '')
                    batch.append((tax_id, name_txt, name_class))
                    if len(batch) >= batch_size:
                        _insert_batch(conn, 'taxonomy_names', batch)
                        batch = []
                if batch:
                    _insert_batch(conn, 'taxonomy_names', batch)
                cursor.execute("COMMIT;")
            except Exception as e:
                logger.error(f"Transaction failed for taxonomy_names: {e}. Rolling back.")
                cursor.execute("ROLLBACK;")
                raise

        # Process merged.dmp
        logger.info("Processing merged.dmp...")
        merged_file = tar.extractfile("merged.dmp")
        if merged_file:
            cursor.execute("BEGIN TRANSACTION;")
            try:
                batch = []
                for line in tqdm(merged_file, desc="Populating taxonomy_merged"):
                    fields = line.decode('utf-8').split('\t|\t')
                    old_tax_id = int(fields[0])
                    new_tax_id = int(fields[1].strip().replace('\t|', ''))
                    batch.append((old_tax_id, new_tax_id))
                    if len(batch) >= batch_size:
                        _insert_batch(conn, 'taxonomy_merged', batch)
                        batch = []
                if batch:
                    _insert_batch(conn, 'taxonomy_merged', batch)
                cursor.execute("COMMIT;")
            except Exception as e:
                logger.error(f"Transaction failed for taxonomy_merged: {e}. Rolling back.")
                cursor.execute("ROLLBACK;")
                raise

    logger.info(f"Finished processing {file_path.name}.")


def verify_database_integrity(conn: sqlite3.Connection):
    """
    Performs a spot check on the database to verify data integrity against source files.
    """
    logger.info("Verifying database integrity...")
    cursor = conn.cursor()
    num_samples = 5

    try:
        # --- Verify pubchem_smiles ---
        logger.info("Checking 'pubchem_smiles' table...")
        cursor.execute(f"SELECT cid, smiles FROM pubchem_smiles ORDER BY RANDOM() LIMIT {num_samples}")
        sample_rows = cursor.fetchall()
        if not sample_rows: raise ValueError("Table is empty.")
        
        sample_cids = {row[0] for row in sample_rows}
        source_data = {}
        with gzip.open(DATA_DIR / "CID-SMILES.gz", 'rt') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                cid = int(parts[0])
                if cid in sample_cids:
                    source_data[cid] = parts[1]
                if len(source_data) == len(sample_cids): break
        
        for cid, smiles in sample_rows:
            if smiles != source_data.get(cid):
                raise ValueError(f"Data mismatch for CID {cid}!")
        logger.info("'pubchem_smiles' check passed.")

        # --- Verify pubchem_inchikey ---
        logger.info("Checking 'pubchem_inchikey' table...")
        cursor.execute(f"SELECT cid, inchikey FROM pubchem_inchikey ORDER BY RANDOM() LIMIT {num_samples}")
        sample_rows = cursor.fetchall()
        if not sample_rows: raise ValueError("Table is empty.")

        sample_cids = {row[0] for row in sample_rows}
        source_data = {}
        with gzip.open(DATA_DIR / "CID-InChI-Key.gz", 'rt') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                cid = int(parts[0])
                if cid in sample_cids:
                    source_data[cid] = parts[2]
                if len(source_data) == len(sample_cids): break
        
        for cid, inchikey in sample_rows:
            if inchikey != source_data.get(cid):
                raise ValueError(f"Data mismatch for CID {cid}!")
        logger.info("'pubchem_inchikey' check passed.")

        # --- Verify pubchem_synonyms ---
        logger.info("Checking 'pubchem_synonyms' table...")
        cursor.execute(f"SELECT cid, synonym FROM pubchem_synonyms ORDER BY RANDOM() LIMIT {num_samples}")
        sample_rows = cursor.fetchall()
        if not sample_rows: raise ValueError("Table is empty.")

        sample_pairs = set(sample_rows)
        found_pairs = set()
        with gzip.open(DATA_DIR / "CID-Synonym-filtered.gz", 'rt') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                pair = (int(parts[0]), parts[1])
                if pair in sample_pairs:
                    found_pairs.add(pair)
                if len(found_pairs) == len(sample_pairs): break
        
        if found_pairs != sample_pairs:
            raise ValueError(f"Mismatch found! Missing pairs: {sample_pairs - found_pairs}")
        logger.info("'pubchem_synonyms' check passed.")

        # --- Verify taxonomy tables ---
        with tarfile.open(DATA_DIR / "taxdump.tar.gz", "r:gz") as tar:
            # Verify taxonomy_nodes
            logger.info("Checking 'taxonomy_nodes' table...")
            cursor.execute(f"SELECT tax_id, parent_tax_id, rank FROM taxonomy_nodes ORDER BY RANDOM() LIMIT {num_samples}")
            sample_rows = cursor.fetchall()
            if not sample_rows: raise ValueError("Table is empty.")
            
            sample_ids = {row[0] for row in sample_rows}
            source_data = {}
            nodes_file = tar.extractfile("nodes.dmp")
            if not nodes_file: raise FileNotFoundError("nodes.dmp not in tar archive")
            for line in nodes_file:
                fields = line.decode('utf-8').split('\t|\t')
                tax_id = int(fields[0])
                if tax_id in sample_ids:
                    source_data[tax_id] = (int(fields[1]), fields[2])
                if len(source_data) == len(sample_ids): break
            
            for tax_id, parent_tax_id, rank in sample_rows:
                if (parent_tax_id, rank) != source_data.get(tax_id):
                    raise ValueError(f"Data mismatch for tax_id {tax_id}!")
            logger.info("'taxonomy_nodes' check passed.")
            
            # Verify taxonomy_names
            logger.info("Checking 'taxonomy_names' table...")
            cursor.execute(f"SELECT tax_id, name, name_class FROM taxonomy_names ORDER BY RANDOM() LIMIT {num_samples}")
            sample_rows = cursor.fetchall()
            if not sample_rows: raise ValueError("Table is empty.")

            sample_pairs = set(sample_rows)
            found_pairs = set()
            names_file = tar.extractfile("names.dmp")
            if not names_file: raise FileNotFoundError("names.dmp not in tar archive")
            for line in names_file:
                fields = line.decode('utf-8').split('\t|\t')
                pair = (int(fields[0]), fields[1], fields[3].strip().replace('\t|', ''))
                if pair in sample_pairs:
                    found_pairs.add(pair)
                if len(found_pairs) == len(sample_pairs): break
            
            if found_pairs != sample_pairs:
                raise ValueError(f"Mismatch found! Missing pairs: {sample_pairs - found_pairs}")
            logger.info("'taxonomy_names' check passed.")

    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        raise

    logger.info("Database integrity verification passed successfully.")


def build_database():
    """
    Main function to build the SQLite database from downloaded reference files.
    It checks each table for data before populating, allowing it to be resumed.
    """
    logger.info("Starting database build process...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Always ensure table schemas exist
        create_tables(conn)

        # --- Process each file and populate tables if they are empty ---

        cursor.execute("SELECT COUNT(*) FROM pubchem_smiles")
        if cursor.fetchone()[0] == 0:
            process_cid_smiles(conn)
        else:
            logger.info("Table 'pubchem_smiles' is already populated. Skipping.")

        cursor.execute("SELECT COUNT(*) FROM pubchem_inchikey")
        if cursor.fetchone()[0] == 0:
            process_cid_inchikey(conn)
        else:
            logger.info("Table 'pubchem_inchikey' is already populated. Skipping.")

        cursor.execute("SELECT COUNT(*) FROM pubchem_synonyms")
        if cursor.fetchone()[0] == 0:
            process_cid_synonyms(conn)
        else:
            logger.info("Table 'pubchem_synonyms' is already populated. Skipping.")

        cursor.execute("SELECT COUNT(*) FROM taxonomy_nodes")
        if cursor.fetchone()[0] == 0:
            # taxdump processing populates multiple tables, so we only need to check one
            process_taxdump(conn)
        else:
            logger.info("Taxonomy tables are already populated. Skipping.")

        # Create indexes after all data is inserted for faster build time
        create_indexes(conn)

        # Final verification step
        verify_database_integrity(conn)

        conn.close()
        logger.info("Database build process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the database build process: {e}", exc_info=True)