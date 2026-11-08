#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, .env, synthetic data and trained model.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "Python 3.10+ required (found: $("$PYTHON" --version))"; exit 1; }

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtualenv in $VENV"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Installing dependencies"
python -m pip install --upgrade pip -q
python -m pip install -e ".[llm,ui,dev]" -q

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example (offline mode, no API key needed)"
  cp .env.example .env
fi

echo "==> Generating synthetic CRM data"
python -m crm_agent.data.generate

echo "==> Training propensity model"
python -m crm_agent.scoring.propensity

echo
echo "Setup complete. Start the app with: ./run.sh   (see ./run.sh --help)"
