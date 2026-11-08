#!/usr/bin/env bash
# Run RM Copilot. Usage: ./run.sh [ui|chat|demo|test|eval]   (default: ui)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "No .venv found - running ./setup.sh first"
  ./setup.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

MODE="${1:-ui}"
case "$MODE" in
  ui)   exec streamlit run app/streamlit_app.py "${@:2}" ;;
  chat) exec python -m crm_agent.interface.cli "${@:2}" ;;
  demo) exec python scripts/demo.py --pause "${@:2}" ;;
  test) exec python -m pytest -q "${@:2}" ;;
  eval) exec python scripts/evaluate.py "${@:2}" ;;
  -h|--help|help)
    echo "Usage: ./run.sh [ui|chat|demo|test|eval]"
    echo "  ui    Streamlit chat UI (default) - http://localhost:8501"
    echo "  chat  Conversational CLI"
    echo "  demo  Scripted walkthrough of the use cases"
    echo "  test  Run the test suite"
    echo "  eval  Model / guardrail / latency report" ;;
  *) echo "Unknown mode: $MODE (try ./run.sh --help)"; exit 1 ;;
esac
