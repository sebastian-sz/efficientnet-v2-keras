lint:
	pre-commit run --all-files

test:
	python -m unittest -f tests/*.py