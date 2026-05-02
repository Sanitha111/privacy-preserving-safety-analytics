# utils/database.py — Ghost-Vision Database Setup
import sqlite3
import os
from config import DB_PATH, EMBEDDING_DB_PATH

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def get_embedding_connection():
    os.makedirs(os.path.dirname(EMBEDDING_DB_PATH), exist_ok=True)
    return sqlite3.connect(EMBEDDING_DB_PATH)

def initialize_db():
    # Main safety database
    conn = get_connection()
    cursor = conn.cursor()

    # Skeleton sequences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skeleton_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            anonymous_id TEXT,
            action_detected TEXT,
            confidence REAL,
            environment TEXT,
            alert_triggered INTEGER DEFAULT 0,
            skeleton_data TEXT
        )
    ''')

    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            anonymous_id TEXT,
            action TEXT,
            confidence REAL,
            environment TEXT,
            resolved INTEGER DEFAULT 0
        )
    ''')

    # Erasure requests table (DPDP Act compliance)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS erasure_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            request_id TEXT,
            records_deleted INTEGER,
            status TEXT
        )
    ''')

    conn.commit()
    conn.close()

    # Privacy/embedding database
    emb_conn = get_embedding_connection()
    emb_cursor = emb_conn.cursor()

    # Face embeddings table — stores math vectors NOT photos
    emb_cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anonymous_id TEXT UNIQUE,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            environment TEXT
        )
    ''')

    emb_conn.commit()
    emb_conn.close()

    print("✅ Ghost-Vision database initialized!")

if __name__ == "__main__":
    initialize_db()