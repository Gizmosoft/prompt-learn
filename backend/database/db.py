'''
    This file handles db connections to SQLite3
'''

import os
import sqlite3
from contextlib import contextmanager
from .constants import SCHEMA_SQL

DB_PATH = os.path.join(os.path.dirname(__file__), "prompt_analyzer.db")

DB_SCHEMA = SCHEMA_SQL

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connect() as conn:
        conn.executescript(DB_SCHEMA)

