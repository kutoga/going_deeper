VENV_DIR=.venv
PYTHON=python3.6
DOCKER_NAME=going_deeper
DOCKER_ENTRYPOINT=

.PHONY: venv docker_build docker_run

venv:
	@[ -d "$(VENV_DIR)" ] || python3.6 -mvenv "$(VENV_DIR)" > /dev/null
	@/bin/bash -c "source $(VENV_DIR)/bin/activate; python3.6 -mpip install -r requirements.txt" > /dev/null
	@echo "source $(VENV_DIR)/bin/activate"

docker_build:
	docker build -t $(DOCKER_NAME) .

docker_run: docker_build
	@if [ -z "$(DOCKER_ENTRYPOINT)"]; then docker run -it --rm $(DOCKER_NAME); else docker run -it --rm --entrypoint "$(DOCKER_ENTRYPOINT)" $(DOCKER_NAME); fi
