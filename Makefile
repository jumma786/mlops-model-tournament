.PHONY: install test tournament mlflow-ui clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

tournament:
	python src/tournament.py --n-samples 5000

tournament-large:
	python src/tournament.py --n-samples 20000 --experiment-name "bank-marketing-large"

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	rm -rf reports/* mlruns/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install        Install dependencies"
	@echo "  make test           Run pytest suite"
	@echo "  make tournament     Run 5-model tournament (5K samples)"
	@echo "  make tournament-large  Run with 20K samples"
	@echo "  make mlflow-ui      Launch MLflow tracking UI"
	@echo "  make clean          Remove generated files"
