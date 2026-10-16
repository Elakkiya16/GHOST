PYTHON ?= python3

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/run_experiments.py

attacks:
	$(PYTHON) scripts/run_attack_suite.py

inversion:
	$(PYTHON) scripts/run_model_inversion.py

check:
	$(PYTHON) -m compileall src scripts
