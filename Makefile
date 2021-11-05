lint:
	pre-commit run --all-files

test:
	@for f in $(shell ls test_efficientnet_v2/test*.py); do \
  		echo $${f};\
		python $${f};\
		done

generate_weight_hashes:
	python scripts/generate_weight_hashes.py \
		--input weights/
