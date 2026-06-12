PYTHON ?= python3

.PHONY: install data train irradiance evaluate app test all

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) run_pipeline.py --step features

train:
	$(PYTHON) run_pipeline.py --step train

irradiance:
	$(PYTHON) run_pipeline.py --step irradiance

evaluate:
	$(PYTHON) run_pipeline.py --step evaluate

app:
	$(PYTHON) -m streamlit run app/dashboard.py

test:
	$(PYTHON) -m pytest -q

all:
	$(PYTHON) run_pipeline.py --step all
