#!/usr/bin/env bash

declare -a namesArray=(
"b0"
"b1"
"b2"
"b3"
"s"
"m"
"l"
"xl"
)

for val1 in ${namesArray[*]}; do
    echo "${val1}"
    # Imagenet 1k
    if [ "${val1}" == "xl" ]; then
      echo "No Imagenet-1k weights for XL variant."
    else
      python scripts/convert_weights.py \
        --model "${val1}" \
        --ckpt weights/original_weights/efficientnetv2-"${val1}" \
        --output weights/efficientnetv2-"${val1}".h5

      python scripts/convert_weights.py \
        --model "${val1}" \
        --include_top=False \
        --ckpt weights/original_weights/efficientnetv2-"${val1}" \
        --output weights/efficientnetv2-"${val1}"_notop.h5
    fi

    # Imagenet 21k
    python scripts/convert_weights.py \
        --model "${val1}" \
        --num_classes 21843 \
        --ckpt weights/original_weights/efficientnetv2-"${val1}"-21k \
        --output weights/efficientnetv2-"${val1}"-21k.h5

      python scripts/convert_weights.py \
        --model "${val1}" \
        --include_top=False \
        --num_classes 21843 \
        --ckpt weights/original_weights/efficientnetv2-"${val1}"-21k \
        --output weights/efficientnetv2-"${val1}"-21k_notop.h5

    # Imagenet 21k-ft1k
    python scripts/convert_weights.py \
        --model "${val1}" \
        --ckpt weights/original_weights/efficientnetv2-"${val1}"-21k-ft1k \
        --output weights/efficientnetv2-"${val1}"-21k-ft1k.h5

      python scripts/convert_weights.py \
        --model "${val1}" \
        --include_top=False \
        --ckpt weights/original_weights/efficientnetv2-"${val1}"-21k-ft1k \
        --output weights/efficientnetv2-"${val1}"-21k-ft1k_notop.h5

  done
