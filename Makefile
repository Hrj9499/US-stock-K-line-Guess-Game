.PHONY: install run play test

install:
	pip install -U pip
	pip install -e .

run:
	uvicorn us_kline_guess.api:app --reload --port 8000

play:
	UKG_DATA_PROVIDER=sample us-kline-guess play --ticker AAPL --hint-level 3

test:
	pytest -q
