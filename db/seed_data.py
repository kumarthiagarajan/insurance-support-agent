import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "insurance.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def seed():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.executescript(
        "DELETE FROM claims; DELETE FROM billing; DELETE FROM policies; DELETE FROM customers;"
    )

    conn.executemany(
        "INSERT INTO customers (customer_id, name, email, phone) VALUES (?, ?, ?, ?)",
        [
            ("CUST001", "Maria Gomez", "maria.gomez@example.com", "555-0101"),
            ("CUST002", "James Chen", "james.chen@example.com", "555-0102"),
        ],
    )

    conn.executemany(
        """INSERT INTO policies
           (policy_id, customer_id, policy_type, status, coverage_summary,
            premium_monthly, start_date, renewal_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "POL-AUTO-001", "CUST001", "auto", "active",
                "Liability $100k/$300k, collision $500 deductible, comprehensive $250 deductible",
                142.50, "2025-03-01", "2026-03-01",
            ),
            (
                "POL-HOME-001", "CUST001", "home", "active",
                "Dwelling $350,000, personal property $175,000, liability $300,000",
                98.00, "2025-01-15", "2026-01-15",
            ),
            (
                "POL-AUTO-002", "CUST002", "auto", "active",
                "Liability $50k/$100k, collision $1000 deductible, comprehensive $500 deductible",
                110.25, "2025-06-10", "2026-06-10",
            ),
        ],
    )

    conn.executemany(
        """INSERT INTO billing
           (invoice_id, customer_id, policy_id, amount_due, due_date, status, payment_method)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            ("INV-1001", "CUST001", "POL-AUTO-001", 142.50, "2026-08-01", "overdue",
             "auto-pay (card ending 4321)"),
            ("INV-1002", "CUST001", "POL-HOME-001", 98.00, "2026-08-15", "due",
             "auto-pay (card ending 4321)"),
            ("INV-1003", "CUST002", "POL-AUTO-002", 110.25, "2026-07-20", "paid",
             "manual (bank transfer)"),
        ],
    )

    conn.executemany(
        """INSERT INTO claims
           (claim_id, customer_id, policy_id, claim_type, status, filed_date,
            description, adjuster_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ("CLM-5001", "CUST001", "POL-AUTO-001", "collision", "under_review", "2026-08-01",
             "Rear-end collision at intersection, minor bumper damage", "Alex Rivera"),
            ("CLM-5002", "CUST002", "POL-AUTO-002", "comprehensive", "approved", "2026-07-10",
             "Windshield crack from road debris", "Priya Nair"),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Seeded database at {DB_PATH}")


if __name__ == "__main__":
    seed()
