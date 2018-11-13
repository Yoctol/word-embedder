.DEFAULT_GOAL := all

.PHONY: install
install:
	pip install pipenv
	pipenv install --dev

.PHONY: lint
lint:
	pipenv run python -m flake8 .

.PHONY: test
test:
	pipenv run python -m unittest -v

.PHONY: all
all: test lint
