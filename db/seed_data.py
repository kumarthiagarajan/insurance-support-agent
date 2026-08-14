import datetime
import sqlite3
from pathlib import Path

import openpyxl

DB_PATH = Path(__file__).parent / "insurance.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SEED_XLSX_PATH = Path(__file__).parent / "Insurance_Support_Agent_Seed_Data.xlsx"

# Maps each DB table to the workbook sheet that seeds it. Column order in each
# sheet's header row must match that table's column order in schema.sql.
TABLE_SHEETS = {
    "customers": "Customers",
    "policies": "Policies",
    "billing": "Billing",
    "claims": "Claims",
}


def _load_rows(wb, sheet_name, text_columns):
    ws = wb[sheet_name]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    columns = [c for c in header if c is not None]
    values = []
    for row in rows:
        if row[0] is None:
            continue
        cell_values = []
        for column, value in zip(columns, row[: len(columns)]):
            if isinstance(value, datetime.datetime):
                value = value.date().isoformat()
            elif isinstance(value, int) and column in text_columns:
                value = str(value)
            cell_values.append(value)
        values.append(tuple(cell_values))
    return columns, values


def seed():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.executescript(
        "DELETE FROM claims; DELETE FROM billing; DELETE FROM policies; DELETE FROM customers;"
    )

    wb = openpyxl.load_workbook(SEED_XLSX_PATH, data_only=True)
    for table, sheet_name in TABLE_SHEETS.items():
        text_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table})")
            if row[2].upper() == "TEXT"
        }
        columns, values = _load_rows(wb, sheet_name, text_columns)
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )

    conn.commit()
    conn.close()
    print(f"Seeded database at {DB_PATH} from {SEED_XLSX_PATH.name}")


if __name__ == "__main__":
    seed()
