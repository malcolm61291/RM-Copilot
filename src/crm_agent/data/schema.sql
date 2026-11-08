-- Synthetic banking CRM schema (SQLite). Mirrors the shape of a core-banking + CRM extract.

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    gender             TEXT,
    age                INTEGER,
    city               TEXT,
    occupation_type    TEXT CHECK (occupation_type IN ('salaried', 'self_employed')),
    segment            TEXT CHECK (segment IN ('mass', 'affluent', 'priority')),
    tenure_months      INTEGER,
    preferred_language TEXT DEFAULT 'en',
    kyc_status         TEXT,
    credit_score       INTEGER,          -- bureau score snapshot
    whatsapp_opt_in    INTEGER DEFAULT 0,
    dnd                INTEGER DEFAULT 0, -- Do-Not-Disturb registry flag
    rm_id              TEXT
);

CREATE TABLE accounts (
    account_id   TEXT PRIMARY KEY,
    customer_id  TEXT REFERENCES customers(customer_id),
    type         TEXT CHECK (type IN ('savings', 'fixed_deposit', 'credit_card', 'personal_loan', 'home_loan')),
    balance      REAL,     -- deposits: balance; credit card: outstanding; loans: principal outstanding
    credit_limit REAL,
    emi          REAL,
    opened_on    TEXT,
    status       TEXT
);

CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    date        TEXT,
    amount      REAL,
    direction   TEXT CHECK (direction IN ('credit', 'debit')),
    category    TEXT,
    channel     TEXT
);
CREATE INDEX idx_txn_customer_date ON transactions(customer_id, date);

-- Digital-banking signals (app / web), e.g. viewed the personal-loan page.
CREATE TABLE digital_events (
    customer_id TEXT REFERENCES customers(customer_id),
    date        TEXT,
    event       TEXT
);
CREATE INDEX idx_events_customer ON digital_events(customer_id, date);

CREATE TABLE products (
    product_id        TEXT PRIMARY KEY,
    name              TEXT,
    product_type      TEXT,
    min_income        REAL,
    min_credit_score  INTEGER,
    max_amount        REAL,
    rate_from         REAL,
    max_tenure_months INTEGER,
    requires          TEXT,  -- simple policy tag: salaried | active_pl | fixed_deposit | none
    pitch             TEXT
);

-- Historical campaign outcomes: training labels for the propensity model.
CREATE TABLE campaign_history (
    customer_id  TEXT REFERENCES customers(customer_id),
    product_type TEXT,
    offered_on   TEXT,
    channel      TEXT,
    converted    INTEGER
);

-- Outbound queue (stands in for a WhatsApp Business API provider).
CREATE TABLE outbox (
    message_id  TEXT PRIMARY KEY,
    customer_id TEXT,
    channel     TEXT,
    body        TEXT,
    status      TEXT,
    approved_by TEXT,
    thread_id   TEXT,
    created_at  TEXT
);

-- Append-only audit trail of agent decisions and human approvals.
CREATE TABLE audit_log (
    ts        TEXT,
    thread_id TEXT,
    actor     TEXT,
    event     TEXT,
    payload   TEXT
);
