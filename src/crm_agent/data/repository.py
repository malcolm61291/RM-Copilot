"""Data-access layer. The only module that knows SQL; tools call these functions.

Swapping SQLite for a core-banking API / warehouse means re-implementing this module only.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from crm_agent.config import settings

BIG_TICKET_SQL = "('travel','education','home_improvement','wedding','medical')"

# One pass over the book: 6-month income, recent big-ticket spend vs baseline, top recent spend
# category (window function), account exposures, digital intent events and campaign history.
FEATURE_SQL = f"""
WITH params AS (SELECT date(:as_of) AS as_of),
income AS (
    SELECT t.customer_id,
           SUM(CASE WHEN t.date >= date(p.as_of, '-90 days') THEN t.amount ELSE 0 END) / 3.0  AS income_3m,
           SUM(CASE WHEN t.date <  date(p.as_of, '-90 days') THEN t.amount ELSE 0 END) / 3.0  AS income_prev_3m,
           COUNT(DISTINCT CAST((julianday(p.as_of) - julianday(t.date)) / 30 AS INTEGER))  AS income_months
    FROM transactions t CROSS JOIN params p
    WHERE t.direction = 'credit' AND t.category IN ('salary', 'business_income')
      AND t.date >= date(p.as_of, '-180 days')
    GROUP BY t.customer_id
),
spend AS (
    SELECT t.customer_id,
           SUM(CASE WHEN t.date >= date(p.as_of, '-45 days') AND t.category IN {BIG_TICKET_SQL}
                    THEN t.amount ELSE 0 END)                                              AS big_ticket_45d,
           SUM(CASE WHEN t.date <  date(p.as_of, '-45 days') AND t.category IN {BIG_TICKET_SQL}
                    THEN t.amount ELSE 0 END) / 3.0                                        AS big_ticket_baseline_45d
    FROM transactions t CROSS JOIN params p
    WHERE t.direction = 'debit' AND t.date >= date(p.as_of, '-180 days')
    GROUP BY t.customer_id
),
recent_category AS (
    SELECT customer_id, category AS recent_big_ticket_category
    FROM (
        SELECT t.customer_id, t.category,
               ROW_NUMBER() OVER (PARTITION BY t.customer_id ORDER BY SUM(t.amount) DESC) AS rn
        FROM transactions t CROSS JOIN params p
        WHERE t.direction = 'debit' AND t.category IN {BIG_TICKET_SQL}
          AND t.date >= date(p.as_of, '-45 days')
        GROUP BY t.customer_id, t.category
    ) WHERE rn = 1
),
exposure AS (
    SELECT customer_id,
           SUM(CASE WHEN type = 'savings'       THEN balance ELSE 0 END)                    AS savings_balance,
           SUM(CASE WHEN type = 'fixed_deposit' THEN balance ELSE 0 END)                    AS fd_balance,
           SUM(CASE WHEN type IN ('personal_loan', 'home_loan') AND status = 'active'
                    THEN emi ELSE 0 END)                                                    AS monthly_emi,
           MAX(CASE WHEN type = 'personal_loan' AND status = 'active' THEN 1 ELSE 0 END)    AS has_active_pl,
           MAX(CASE WHEN type = 'credit_card' THEN 1 ELSE 0 END)                            AS has_credit_card,
           SUM(CASE WHEN type = 'credit_card' THEN balance ELSE 0 END)
             / NULLIF(SUM(CASE WHEN type = 'credit_card' THEN credit_limit END), 0)         AS cc_utilisation
    FROM accounts
    GROUP BY customer_id
),
digital AS (
    SELECT e.customer_id, COUNT(*) AS loan_intent_events_30d
    FROM digital_events e CROSS JOIN params p
    WHERE e.date >= date(p.as_of, '-30 days')
    GROUP BY e.customer_id
),
campaigns AS (
    SELECT customer_id, COUNT(*) AS past_pl_offers, SUM(converted) AS past_pl_conversions
    FROM campaign_history WHERE product_type = 'personal_loan'
    GROUP BY customer_id
)
SELECT c.*,
       COALESCE(i.income_3m, 0)                 AS monthly_income,
       COALESCE(i.income_prev_3m, 0)            AS monthly_income_prev,
       COALESCE(i.income_months, 0)             AS income_months,
       COALESCE(s.big_ticket_45d, 0)            AS big_ticket_45d,
       COALESCE(s.big_ticket_baseline_45d, 0)   AS big_ticket_baseline_45d,
       rc.recent_big_ticket_category,
       COALESCE(x.savings_balance, 0)           AS savings_balance,
       COALESCE(x.fd_balance, 0)                AS fd_balance,
       COALESCE(x.monthly_emi, 0)               AS monthly_emi,
       COALESCE(x.has_active_pl, 0)             AS has_active_pl,
       COALESCE(x.has_credit_card, 0)           AS has_credit_card,
       COALESCE(x.cc_utilisation, 0)            AS cc_utilisation,
       COALESCE(d.loan_intent_events_30d, 0)    AS loan_intent_events_30d,
       COALESCE(cp.past_pl_offers, 0)           AS past_pl_offers,
       COALESCE(cp.past_pl_conversions, 0)      AS past_pl_conversions
FROM customers c
LEFT JOIN income i           ON i.customer_id  = c.customer_id
LEFT JOIN spend s            ON s.customer_id  = c.customer_id
LEFT JOIN recent_category rc ON rc.customer_id = c.customer_id
LEFT JOIN exposure x         ON x.customer_id  = c.customer_id
LEFT JOIN digital d          ON d.customer_id  = c.customer_id
LEFT JOIN campaigns cp       ON cp.customer_id = c.customer_id
"""


class Repository:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Run `make data` (python -m crm_agent.data.generate)."
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---- reads -------------------------------------------------------------------------
    def as_of(self) -> str:
        with self._conn() as con:
            return con.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()[0]

    def find_customers(self, filters: dict | None = None) -> list[dict]:
        """Customer master lookup with whitelisted filters (no free-form SQL from the LLM)."""
        filters = filters or {}
        clauses, params = ["rm_id = :rm_id"], {"rm_id": settings.rm_id}
        allowed = {"city": "city", "segment": "segment", "occupation_type": "occupation_type",
                   "preferred_language": "preferred_language"}
        for key, column in allowed.items():
            value = filters.get(key)
            if value:
                values = value if isinstance(value, list) else [value]
                names = [f"{key}_{n}" for n in range(len(values))]
                clauses.append(f"LOWER({column}) IN ({', '.join(':' + n for n in names)})")
                params.update({n: str(v).lower() for n, v in zip(names, values)})
        if filters.get("min_age"):
            clauses.append("age >= :min_age"); params["min_age"] = int(filters["min_age"])
        if filters.get("max_age"):
            clauses.append("age <= :max_age"); params["max_age"] = int(filters["max_age"])
        if filters.get("customer_ids"):
            ids = filters["customer_ids"]
            names = [f"id_{n}" for n in range(len(ids))]
            clauses.append(f"customer_id IN ({', '.join(':' + n for n in names)})")
            params.update(dict(zip(names, ids)))
        sql = f"SELECT * FROM customers WHERE {' AND '.join(clauses)} ORDER BY customer_id"
        with self._conn() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def search_customer(self, text: str) -> list[dict]:
        text = text.strip()
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM customers WHERE UPPER(customer_id) = UPPER(?) OR LOWER(name) LIKE LOWER(?) "
                "ORDER BY customer_id LIMIT 5",
                (text, f"%{text}%"),
            ).fetchall()
        return [dict(r) for r in rows]

    def customer_features(self, customer_ids: list[str] | None = None) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query(FEATURE_SQL, con, params={"as_of": self.as_of()})
        if customer_ids is not None:
            df = df[df["customer_id"].isin(customer_ids)]
        return df.reset_index(drop=True)

    def products(self, product_type: str | None = None) -> list[dict]:
        with self._conn() as con:
            if product_type:
                rows = con.execute("SELECT * FROM products WHERE product_type = ?", (product_type,)).fetchall()
            else:
                rows = con.execute("SELECT * FROM products").fetchall()
        return [dict(r) for r in rows]

    def training_frame(self, product_type: str = "personal_loan") -> pd.DataFrame:
        """Customer features joined to 'did the customer ever convert on this product's campaigns'."""
        feats = self.customer_features()
        with self._conn() as con:
            labels = pd.read_sql_query(
                "SELECT customer_id, MAX(converted) AS converted FROM campaign_history "
                "WHERE product_type = ? GROUP BY customer_id", con, params=(product_type,))
        return feats.merge(labels, on="customer_id", how="inner")

    def outbox(self, thread_id: str | None = None) -> list[dict]:
        with self._conn() as con:
            if thread_id:
                rows = con.execute("SELECT * FROM outbox WHERE thread_id = ? ORDER BY created_at", (thread_id,))
            else:
                rows = con.execute("SELECT * FROM outbox ORDER BY created_at")
            return [dict(r) for r in rows.fetchall()]

    # ---- writes ------------------------------------------------------------------------
    def enqueue_message(self, customer_id: str, channel: str, body: str, approved_by: str, thread_id: str) -> str:
        message_id = f"MSG-{uuid.uuid4().hex[:10].upper()}"
        with self._conn() as con:
            con.execute(
                "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",
                (message_id, customer_id, channel, body, "queued", approved_by, thread_id,
                 datetime.now(timezone.utc).isoformat()),
            )
        return message_id

    def already_sent(self, customer_id: str, body: str) -> bool:
        with self._conn() as con:
            row = con.execute("SELECT 1 FROM outbox WHERE customer_id = ? AND body = ?", (customer_id, body)).fetchone()
        return row is not None

    def audit(self, thread_id: str, actor: str, event: str, payload: dict) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO audit_log VALUES (?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), thread_id, actor, event, json.dumps(payload, default=str)),
            )
