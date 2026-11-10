"""Central configuration. Everything tunable lives here or in environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader so the project has no hard dependency on python-dotenv."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Storage
    db_path: Path = Path(os.getenv("CRM_DB_PATH", str(ROOT / "data" / "crm.db")))
    model_dir: Path = Path(os.getenv("CRM_MODEL_DIR", str(ROOT / "data" / "models")))
    runs_dir: Path = Path(os.getenv("CRM_RUNS_DIR", str(ROOT / "runs")))

    # LLM provider: offline | anthropic | openai
    llm_provider: str = os.getenv("LLM_PROVIDER", "offline").lower()
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))

    # Bank / RM identity (fictional)
    bank_name: str = os.getenv("BANK_NAME", "NovaBank")
    rm_name: str = os.getenv("RM_NAME", "Rahul")
    rm_id: str = os.getenv("RM_ID", "RM001")

    # Web sign-in (demo account; replace with SSO for a real deployment)
    demo_username: str = os.getenv("DEMO_USERNAME", "rahul.rm")
    demo_password: str = os.getenv("DEMO_PASSWORD", "NovaBank@2026")
    show_demo_credentials: bool = os.getenv("SHOW_DEMO_CREDENTIALS", "true").lower() == "true"
    session_timeout_minutes: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

    # Business thresholds
    default_top_n: int = 10
    propensity_threshold: float = 0.50
    high_value_quantile: float = 0.70  # top 30% of the book by value score
    priority_weight_propensity: float = 0.65
    priority_weight_value: float = 0.35
    max_revision_rounds: int = 2
    whatsapp_max_chars: int = 700

    # Credit policy
    max_dti_after_loan: float = 0.50  # total EMIs / income after the new loan
    income_multiplier: dict = field(
        default_factory=lambda: {"salaried": 20, "self_employed": 12}
    )


settings = Settings()
