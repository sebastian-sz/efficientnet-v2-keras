#!/bin/bash
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

DOWNLOAD_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-"

for val1 in ${namesArray[*]}; do
    echo "${val1}"
    # Imagenet 1k
    if [ "${val1}" == "xl" ]; then
      echo "No Imagenet-1k weights for XL variant."
    else
      curl "${DOWNLOAD_URL}""${val1}".tgz | tar xz -C weights/original_weights/
    fi

    # Imagenet 21k
    curl "${DOWNLOAD_URL}""${val1}"-21k.tgz | tar xz -C weights/original_weights/

    # Imagenet 21k-ft1k
    curl "${DOWNLOAD_URL}""${val1}"-21k-ft1k.tgz | tar xz -C weights/original_weights/

  done
