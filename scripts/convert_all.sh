#!/usr/bin/env bash

####
# Convert weights pretrained on Imagenet 1k
####

# With top
python scripts/convert_weights.py \
    --model b0 \
    --ckpt weights/original_weights/efficientnetv2-b0 \
    --output weights/efficientnetv2-b0.h5

python scripts/convert_weights.py \
    --model b1 \
    --ckpt weights/original_weights/efficientnetv2-b1 \
    --output weights/efficientnetv2-b1.h5 \

python scripts/convert_weights.py \
    --model b2 \
    --ckpt weights/original_weights/efficientnetv2-b2 \
    --output weights/efficientnetv2-b2.h5

python scripts/convert_weights.py \
    --model b3 \
    --ckpt weights/original_weights/efficientnetv2-b3 \
    --output weights/efficientnetv2-b3.h5

python scripts/convert_weights.py \
    --model s \
    --ckpt weights/original_weights/efficientnetv2-s \
    --output weights/efficientnetv2-s.h5

python scripts/convert_weights.py \
    --model m \
    --ckpt weights/original_weights/efficientnetv2-m \
    --output weights/efficientnetv2-m.h5

python scripts/convert_weights.py \
    --model l \
    --ckpt weights/original_weights/efficientnetv2-l \
    --output weights/efficientnetv2-l.h5

# Without top
python scripts/convert_weights.py \
    --model b0 \
    --ckpt weights/original_weights/efficientnetv2-b0 \
    --output weights/efficientnetv2-b0_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model b1 \
    --ckpt weights/original_weights/efficientnetv2-b1 \
    --output weights/efficientnetv2-b1_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model b2 \
    --ckpt weights/original_weights/efficientnetv2-b2 \
    --output weights/efficientnetv2-b2_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model b3 \
    --ckpt weights/original_weights/efficientnetv2-b3 \
    --output weights/efficientnetv2-b3_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model s \
    --ckpt weights/original_weights/efficientnetv2-s \
    --output weights/efficientnetv2-s_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model m \
    --ckpt weights/original_weights/efficientnetv2-m \
    --output weights/efficientnetv2-m_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model l \
    --ckpt weights/original_weights/efficientnetv2-l \
    --output weights/efficientnetv2-l_notop.h5 \
    --include_top=False

#####
# Convert pretrained on imagenet 21k and fine tuned on 1k
####

# Top
python scripts/convert_weights.py \
    --model s \
    --ckpt weights/original_weights/efficientnetv2-s-21k-ft1k \
    --output weights/efficientnetv2-s-21k-ft1k.h5

python scripts/convert_weights.py \
    --model m \
    --ckpt weights/original_weights/efficientnetv2-m-21k-ft1k \
    --output weights/efficientnetv2-m-21k-ft1k.h5

python scripts/convert_weights.py \
    --model l \
    --ckpt weights/original_weights/efficientnetv2-l-21k-ft1k \
    --output weights/efficientnetv2-l-21k-ft1k.h5

python scripts/convert_weights.py \
    --model xl \
    --ckpt weights/original_weights/efficientnetv2-xl-21k-ft1k \
    --output weights/efficientnetv2-xl-21k-ft1k.h5

# Notop
python scripts/convert_weights.py \
    --model s \
    --ckpt weights/original_weights/efficientnetv2-s-21k-ft1k \
    --output weights/efficientnetv2-s-21k-ft1k_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model m \
    --ckpt weights/original_weights/efficientnetv2-m-21k-ft1k \
    --output weights/efficientnetv2-m-21k-ft1k_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model l \
    --ckpt weights/original_weights/efficientnetv2-l-21k-ft1k \
    --output weights/efficientnetv2-l-21k-ft1k_notop.h5 \
    --include_top=False

python scripts/convert_weights.py \
    --model xl \
    --ckpt weights/original_weights/efficientnetv2-xl-21k-ft1k \
    --output weights/efficientnetv2-xl-21k-ft1k_notop.h5 \
    --include_top=False
