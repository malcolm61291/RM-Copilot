"""Sign-in: password handling, lockout, and the Streamlit sign-in gate end to end."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from crm_agent.auth import MAX_ATTEMPTS, LoginThrottle, authenticate, check_password, hash_password
from crm_agent.config import settings
from crm_agent.formatting import plain_dashes

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
DASHES = ("—", "–")  # em dash, en dash


def test_password_hash_round_trip():
    stored = hash_password("s3cret!")
    assert "s3cret" not in stored
    assert check_password("s3cret!", stored) and not check_password("s3cret?", stored)
    assert hash_password("same") != hash_password("same")          # salted


def test_authenticate_demo_account():
    user = authenticate(settings.demo_username.upper(), settings.demo_password)
    assert user and user.rm_id == settings.rm_id
    assert authenticate(settings.demo_username, "wrong") is None
    assert authenticate("nobody", settings.demo_password) is None


def test_lockout_after_repeated_failures():
    throttle = LoginThrottle()
    for _ in range(MAX_ATTEMPTS):
        throttle.record_failure(now=1000.0)
    assert throttle.seconds_locked(now=1001.0) > 0
    assert throttle.seconds_locked(now=2000.0) == 0


def test_plain_dashes():
    assert plain_dashes("age 21–58") == "age 21 to 58"
    assert plain_dashes("Hi Asha — good news") == "Hi Asha, good news"
    assert plain_dashes("– Rahul, NovaBank") == "Rahul, NovaBank"


def _visible_text(at: AppTest) -> list[str]:
    texts = []
    for kind in ("markdown", "caption", "error", "warning", "info", "success", "subheader"):
        texts += [str(el.value) for el in getattr(at, kind)]
    texts += [b.label for b in at.button] + [t.label for t in at.tabs]
    texts += [f"{m.label} {m.value}" for m in at.metric]
    for df in at.dataframe:
        texts += df.value.astype(str).to_numpy().ravel().tolist() + list(map(str, df.value.columns))
    return texts


def _sign_in(at: AppTest, password: str) -> AppTest:
    at.text_input(key="login_user").input(settings.demo_username)
    at.text_input(key="login_pass").input(password)
    return at.button[0].click().run()     # the form's "Sign in" button is the first button


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_workspace_is_hidden_until_signed_in(app):
    assert not app.exception
    assert not app.tabs and not app.chat_input                 # nothing but the sign-in form
    _sign_in(app, "wrong-password")
    assert any("Incorrect username or password" in e.value for e in app.error)
    assert not app.tabs


def test_demo_button_signs_in_and_sign_out_returns_to_login(app):
    next(b for b in app.button if b.label == "Fill in demo account").click().run()
    assert app.text_input(key="login_pass").value == settings.demo_password
    app.button[0].click().run()
    assert not app.exception and len(app.tabs) == 4
    app.button(key="sign-out").click().run()
    assert not app.tabs and not app.chat_input


def test_no_dashes_anywhere_on_screen(app):
    _sign_in(app, settings.demo_password)
    for command in ("Find high-value customers likely to convert for a personal loan this month "
                    "and generate personalised WhatsApp messages",
                    "Why is #1 on the list?", "edit 2: make it shorter", "approve 1", "reject 3",
                    "Only salaried customers in Mumbai, top 5, in Hindi"):
        app.chat_input[0].set_value(command).run()
        assert not app.exception
        offending = [t for t in _visible_text(app) if any(d in t for d in DASHES)]
        assert not offending, f"dash on screen after {command!r}: {offending[:3]}"
