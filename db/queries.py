import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "insurance.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_customer(customer_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return dict(row) if row else None


def get_policies(customer_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM policies WHERE customer_id = ?", (customer_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_billing(customer_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM billing WHERE customer_id = ?", (customer_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_claims(customer_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE customer_id = ?", (customer_id,)
        ).fetchall()
        return [dict(r) for r in rows]
