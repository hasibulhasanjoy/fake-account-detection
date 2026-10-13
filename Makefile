PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help install train predict serve serve-prod

help:
	@echo "Fake Account Detection - available targets:"
	@echo "  make install     Install all dependencies (requirements.txt + package)"
	@echo "  make train       Run the full training pipeline (ingest, transform, tune, save artifacts)"
	@echo "  make predict     Run the prediction pipeline on dummy input data"
	@echo "  make serve       Start the FastAPI server with auto-reload (dev mode)"
	@echo "  make serve-prod  Start the FastAPI server without auto-reload (prod mode)"
	@echo ""
	@echo "Variables: HOST (default 0.0.0.0), PORT (default 8000), PYTHON (default python)"
	@echo "Example: make serve PORT=9000"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) train.py

predict:
	$(PYTHON) train.py --predict

serve:
	$(PYTHON) -m uvicorn src.api.main:app --reload --host $(HOST) --port $(PORT)

serve-prod:
	$(PYTHON) -m uvicorn src.api.main:app --host $(HOST) --port $(PORT)
