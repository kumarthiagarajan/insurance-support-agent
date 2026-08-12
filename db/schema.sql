CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    policy_type TEXT NOT NULL,
    status TEXT NOT NULL,
    coverage_summary TEXT,
    premium_monthly REAL,
    start_date TEXT,
    renewal_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS billing (
    invoice_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    policy_id TEXT,
    amount_due REAL,
    due_date TEXT,
    status TEXT,
    payment_method TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    policy_id TEXT,
    claim_type TEXT,
    status TEXT,
    filed_date TEXT,
    description TEXT,
    adjuster_name TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
