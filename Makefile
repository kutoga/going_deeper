VENV_DIR=.venv
PYTHON=python3.6

.PHONY: venv

venv:
	@[ -d "$(VENV_DIR)" ] || python3.6 -mvenv "$(VENV_DIR)"
	@/bin/bash -c "source $(VENV_DIR)/bin/activate; python3.6 -mpip install -r requirements.txt"
	@echo "source $(VENV_DIR)/bin/activate"

