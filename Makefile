PY ?= python

.PHONY: setup data train chat demo ui test eval all

setup:            ## install the package and all extras
	$(PY) -m pip install -e ".[llm,ui,dev]"

data:             ## generate the synthetic CRM database
	$(PY) -m crm_agent.data.generate

train: data       ## train the personal-loan propensity model
	$(PY) -m crm_agent.scoring.propensity

chat:             ## conversational CLI
	$(PY) -m crm_agent.interface.cli

demo:             ## scripted walkthrough of the use cases
	$(PY) scripts/demo.py --pause

ui:               ## Streamlit chat UI
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

eval:
	$(PY) scripts/evaluate.py

all: setup train test
