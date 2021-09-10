lint:
	pre-commit run --all-files

test:
	python -m unittest -f tests/*.py

generate_weight_hashes:
	python scripts/generate_weight_hashes.py \
		--input weights/
