lint:
	pre-commit run --all-files

test:
	pytest -x test_efficientnet_v2/test_*  # Run all tests except check_output_consistency.py

generate_weight_hashes:
	python scripts/generate_weight_hashes.py \
		--input weights/
