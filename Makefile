lint:
	pre-commit run --all-files

test:
	python -m unittest -v -f test_efficientnet_v2/test*.py

generate_weight_hashes:
	python scripts/generate_weight_hashes.py \
		--input weights/
