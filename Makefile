.PHONY: start
start:
	uvicorn main:app --reload --port 8989

.PHONY: format
format:
	black .
	isort .