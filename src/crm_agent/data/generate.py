"""Generate a reproducible synthetic banking CRM dataset in SQLite.

Each customer gets a hidden "borrowing need" that drives both their recent behaviour
(big-ticket spends, loan-page visits) and their historical campaign conversions. The
propensity model therefore has a real, learnable signal, and nothing downstream is
hard-coded: every output is computed from these tables.

Usage:  python -m crm_agent.data.generate [--customers 500] [--seed 42] [--as-of 2026-11-08]
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from crm_agent.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

FIRST_M = ["Aarav", "Rohan", "Vikram", "Arjun", "Karan", "Siddharth", "Rahul", "Aditya", "Nikhil",
           "Imran", "Joseph", "Manish", "Suresh", "Pranav", "Harsh", "Farhan", "Kunal", "Vivek",
           "Anil", "Deepak", "Gaurav", "Mohit", "Sameer", "Tushar", "Yash"]
FIRST_F = ["Priya", "Ananya", "Sneha", "Kavya", "Neha", "Pooja", "Ritu", "Aisha", "Meera", "Divya",
           "Sana", "Shreya", "Ishita", "Nisha", "Fatima", "Riya", "Tanvi", "Swati", "Lakshmi", "Maria",
           "Anjali", "Payal", "Zoya", "Bhavna", "Simran"]
LAST = ["Sharma", "Patel", "Iyer", "Nair", "Reddy", "Mehta", "Desai", "Khan", "Fernandes", "Gupta",
        "Joshi", "Kulkarni", "Rao", "Singh", "Shah", "Menon", "Pillai", "Chopra", "D'Souza", "Bose",
        "Kapoor", "Verma", "Malhotra", "Shetty", "Banerjee"]
CITIES = ["Mumbai", "Pune", "Bengaluru", "Delhi", "Hyderabad", "Chennai"]
BIG_TICKET = ["travel", "education", "home_improvement", "wedding", "medical"]

PRODUCTS = [
    # id, name, type, min_income, min_score, max_amount, rate_from, max_tenure, requires, pitch
    ("PL_PREAPPROVED", "Pre-approved Personal Loan", "personal_loan", 50000, 750, 2_500_000, 10.75, 60,
     "salaried", "Minimal paperwork and quick disbursal to your salary account."),
    ("PL_TOPUP", "Personal Loan Top-up", "personal_loan", 30000, 720, 1_000_000, 11.00, 48,
     "active_pl", "Top up your existing loan and keep a single, simple EMI."),
    ("PL_STANDARD", "Personal Loan", "personal_loan", 25000, 700, 1_500_000, 11.50, 60,
     "none", "Flexible tenures to match your cash flows."),
    ("CC_REWARDS", "Rewards Credit Card", "credit_card", 30000, 720, 500_000, 0.0, 0,
     "none", "Reward points on every spend and complimentary lounge access."),
    ("LAFD", "Loan against Fixed Deposit", "loan_against_fd", 0, 0, 5_000_000, 8.50, 36,
     "fixed_deposit", "Unlock funds without breaking your FD, and keep earning interest."),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _round(x: float, base: int = 100) -> float:
    return float(round(x / base) * base)


def generate(db_path: Path, n_customers: int = 1200, seed: int = 42, as_of: date | None = None) -> dict:
    rng = random.Random(seed)
    as_of = as_of or date.today()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_PATH.read_text())
    con.execute("INSERT INTO meta VALUES ('as_of', ?)", (as_of.isoformat(),))
    con.execute("INSERT INTO meta VALUES ('seed', ?)", (str(seed),))
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?)", PRODUCTS)

    customers, accounts, txns, events, campaigns = [], [], [], [], []
    acct_seq = txn_seq = 0

    def add_txn(cid: str, d: date, amount: float, direction: str, category: str, channel: str = "upi") -> None:
        nonlocal txn_seq
        txn_seq += 1
        txns.append((f"T{txn_seq:07d}", cid, d.isoformat(), round(amount, 2), direction, category, channel))

    def add_acct(cid: str, typ: str, balance: float, limit: float = 0, emi: float = 0, status: str = "active") -> None:
        nonlocal acct_seq
        acct_seq += 1
        opened = as_of - timedelta(days=rng.randint(60, 2500))
        accounts.append((f"A{acct_seq:06d}", cid, typ, round(balance, 2), limit, emi, opened.isoformat(), status))

    for i in range(1, n_customers + 1):
        cid = f"C{i:04d}"
        gender = rng.choice("MF")
        name = f"{rng.choice(FIRST_M if gender == 'M' else FIRST_F)} {rng.choice(LAST)}"
        age = rng.randint(22, 64)
        occupation = "salaried" if rng.random() < 0.65 else "self_employed"
        income = _round(min(max(rng.lognormvariate(math.log(65000), 0.6), 18000), 600000))
        credit_score = int(min(max(rng.gauss(735, 55), 560), 880))
        tenure = rng.randint(3, 180)
        city = rng.choice(CITIES)
        lang = "hi" if (city == "Delhi" and rng.random() < 0.6) or rng.random() < 0.12 else "en"
        kyc = "verified" if rng.random() < 0.95 else "pending"
        opt_in = 1 if rng.random() < 0.82 else 0
        dnd = 1 if rng.random() < 0.06 else 0

        # Hidden borrowing need: drives recent signals AND historical conversions.
        need = rng.random()
        intent_cat = rng.choice(BIG_TICKET) if need > 0.55 else None

        savings = _round(income * rng.uniform(0.5, 9))
        has_fd = rng.random() < 0.30
        fd = _round(income * rng.uniform(2, 20), 1000) if has_fd else 0.0
        has_cc = rng.random() < 0.55
        cc_limit = _round(income * rng.uniform(1, 3), 1000) if has_cc else 0.0
        cc_util = rng.uniform(0.05, 0.95) if has_cc else 0.0
        has_pl = rng.random() < 0.15
        pl_emi = _round(income * rng.uniform(0.06, 0.18)) if has_pl else 0.0
        has_hl = rng.random() < 0.12
        hl_emi = _round(income * rng.uniform(0.15, 0.35)) if has_hl else 0.0

        rel_value = savings + fd
        segment = ("priority" if income >= 200000 or rel_value >= 2_500_000
                   else "affluent" if income >= 90000 or rel_value >= 800_000 else "mass")

        customers.append((cid, name, gender, age, city, occupation, segment, tenure, lang, kyc,
                          credit_score, opt_in, dnd, settings.rm_id))

        add_acct(cid, "savings", savings)
        if has_fd:
            add_acct(cid, "fixed_deposit", fd)
        if has_cc:
            add_acct(cid, "credit_card", cc_limit * cc_util, limit=cc_limit)
        if has_pl:
            add_acct(cid, "personal_loan", pl_emi * rng.randint(10, 40), emi=pl_emi)
        if has_hl:
            add_acct(cid, "home_loan", hl_emi * rng.randint(60, 200), emi=hl_emi)

        # Six months of transactions.
        growth = rng.uniform(-0.05, 0.12)
        for m in range(6, 0, -1):
            month_start = as_of - timedelta(days=30 * m)
            month_income = income * (1 + growth * (6 - m) / 6)
            if occupation == "salaried":
                # Occasional missed / delayed salary for a small share of customers.
                if rng.random() > 0.04:
                    add_txn(cid, month_start + timedelta(days=rng.randint(0, 4)),
                            month_income * rng.uniform(0.98, 1.02), "credit", "salary", "neft")
            else:
                parts = rng.randint(2, 5)
                total = month_income * rng.uniform(0.6, 1.4)
                for _ in range(parts):
                    add_txn(cid, month_start + timedelta(days=rng.randint(0, 28)),
                            total / parts, "credit", "business_income", "imps")
            for cat, share in (("rent", 0.22), ("utilities", 0.05), ("groceries", 0.12), ("shopping", 0.10)):
                if rng.random() < 0.9:
                    add_txn(cid, month_start + timedelta(days=rng.randint(3, 27)),
                            month_income * share * rng.uniform(0.6, 1.4), "debit", cat)
            if pl_emi + hl_emi:
                add_txn(cid, month_start + timedelta(days=5), pl_emi + hl_emi, "debit", "emi", "nach")
            # Background big-ticket noise for everyone.
            if rng.random() < 0.12:
                add_txn(cid, month_start + timedelta(days=rng.randint(0, 28)),
                        month_income * rng.uniform(0.1, 0.5), "debit", rng.choice(BIG_TICKET), "card")

        # Recent intent spike (last 45 days) for customers with a real need.
        if intent_cat and rng.random() < 0.8:
            for _ in range(rng.randint(1, 3)):
                add_txn(cid, as_of - timedelta(days=rng.randint(1, 44)),
                        income * rng.uniform(0.3, 1.5) * (0.5 + need), "debit", intent_cat, "card")

        # Digital-banking loan-intent events in the last 30 days.
        if (need > 0.55 and rng.random() < 0.7) or rng.random() < 0.06:
            for _ in range(rng.randint(1, 4)):
                events.append((cid, (as_of - timedelta(days=rng.randint(0, 29))).isoformat(),
                               rng.choice(["viewed_pl_page", "used_emi_calculator", "started_pl_application"])))

        # Historical personal-loan campaigns (labels for the propensity model).
        if rng.random() < 0.65:
            dti = (pl_emi + hl_emi) / income
            logit = (-3.4 + 4.2 * need + 0.7 * (credit_score >= 750) - 1.2 * (dti > 0.40)
                     + 0.4 * (occupation == "salaried") + 0.4 * (segment != "mass") - 0.5 * has_pl
                     + rng.gauss(0, 0.6))
            for _ in range(rng.randint(1, 2)):
                offered = as_of - timedelta(days=rng.randint(60, 540))
                campaigns.append((cid, "personal_loan", offered.isoformat(),
                                  rng.choice(["whatsapp", "sms", "email", "call"]),
                                  int(rng.random() < _sigmoid(logit))))
        # A few credit-card campaigns too (not used for training here; shows multi-product history).
        if rng.random() < 0.2:
            campaigns.append((cid, "credit_card", (as_of - timedelta(days=rng.randint(60, 540))).isoformat(),
                              "email", int(rng.random() < 0.15)))

    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", customers)
    con.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)", accounts)
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?)", txns)
    con.executemany("INSERT INTO digital_events VALUES (?,?,?)", events)
    con.executemany("INSERT INTO campaign_history VALUES (?,?,?,?,?)", campaigns)
    con.commit()
    con.close()
    return {"customers": len(customers), "accounts": len(accounts), "transactions": len(txns),
            "digital_events": len(events), "campaigns": len(campaigns), "as_of": as_of.isoformat()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--customers", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--as-of", type=date.fromisoformat, default=None, help="Reference date (default: today)")
    p.add_argument("--db", type=Path, default=settings.db_path)
    args = p.parse_args()
    stats = generate(args.db, args.customers, args.seed, args.as_of)
    print(f"Generated {args.db}: {stats}")


if __name__ == "__main__":
    main()
