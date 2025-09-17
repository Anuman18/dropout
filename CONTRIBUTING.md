# Contributing

Thanks for your interest in contributing! Please follow the steps below:

1. Fork the repository and create a feature branch:
	- `git checkout -b feat/your-feature`
2. Set up dev environment:
	- `python -m venv .venv && source .venv/bin/activate`
	- `pip install -r requirements.txt`
3. Run locally:
	- `python -m src.generate_dataset --rows 2000 --out data/students.csv`
	- `python -m src.train_model --data data/students.csv`
	- `uvicorn src.api:app --reload --port 8000`
4. Add tests or manual checks and ensure no linter errors.
5. Commit with conventional messages and open a Pull Request.

## Code style
- Python 3.11, type hints preferred.
- Keep changes focused; update README/docs when needed.

## Security
- Don’t commit secrets. Use environment variables.
