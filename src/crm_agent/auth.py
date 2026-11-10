"""Sign-in for the web workspace.

Users come from the environment (see .env.example). Passwords are kept only as salted
PBKDF2-SHA256 hashes and compared in constant time. This is enough for a demo deployment;
a production rollout would delegate to the bank's identity provider (SSO / LDAP) instead.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass

from crm_agent.config import settings

ITERATIONS = 200_000
MAX_ATTEMPTS = 5          # failed sign-ins allowed before a temporary lockout
LOCKOUT_SECONDS = 60


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    role: str
    rm_id: str


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


def _load_users() -> dict[str, tuple[User, str]]:
    user = User(username=settings.demo_username.lower(), display_name=settings.rm_name,
                role="Relationship Manager", rm_id=settings.rm_id)
    return {user.username: (user, hash_password(settings.demo_password))}


_USERS = _load_users()


def authenticate(username: str, password: str) -> User | None:
    record = _USERS.get(username.strip().lower())
    if record is None:
        # Hash anyway so a wrong username takes as long as a wrong password.
        check_password(password, hash_password("not-a-user"))
        return None
    user, stored = record
    return user if check_password(password, stored) else None


@dataclass
class LoginThrottle:
    """Per-session counter of failed attempts, with a short lockout after too many."""
    failures: int = 0
    locked_until: float = 0.0

    def seconds_locked(self, now: float | None = None) -> int:
        return max(0, round(self.locked_until - (now or time.time())))

    def record_failure(self, now: float | None = None) -> None:
        self.failures += 1
        if self.failures >= MAX_ATTEMPTS:
            self.locked_until = (now or time.time()) + LOCKOUT_SECONDS
            self.failures = 0

    def reset(self) -> None:
        self.failures, self.locked_until = 0, 0.0
