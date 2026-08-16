# Convenience targets for the PI-PPO reproducibility repository.

.PHONY: help install verify reproduce figures clean

help:
	@echo "Targets:"
	@echo "  make install    - install Python dependencies"
	@echo "  make verify     - quick reproducibility check (~2 min)"
	@echo "  make reproduce  - full run producing the paper's numbers (3 seeds x 1000 ep)"
	@echo "  make figures    - regenerate figures from converged results"
	@echo "  make clean      - remove generated outputs"

install:
	pip install -r requirements.txt

verify:
	python tests/verify.py

reproduce:
	python src/simulator.py --full

figures:
	python scripts/make_figures.py

clean:
	rm -f src/results_generated.json src/results_generated.md
	rm -rf src/__pycache__ tests/__pycache__
