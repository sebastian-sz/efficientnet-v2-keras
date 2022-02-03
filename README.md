EfficientNetV2 models rewritten in Keras functional API.

### Changelog:
* Feb 2022: 
  * As of 2.8 Tensorflow release, the models in this repository (apart from XL variant) are accessible through `keras.applications.efficientnet_v2`  
You are free to use this repo or Keras directly.
* Nov 2021: 
  * added more weights variants from original repo.
  * added option to manually get preprocessing layer.
* Sept. 2021 - Added XL model variant.
    * Changed layer naming convention.
    * Re-exported weights.


# Table of contens
1. [Introduction](https://github.com/sebastian-sz/efficientnet-v2-keras#introduction)
2. [Quickstart](https://github.com/sebastian-sz/efficientnet-v2-keras#quickstart)
3. [Installation](https://github.com/sebastian-sz/efficientnet-v2-keras#installation)
4. [How to use](https://github.com/sebastian-sz/efficientnet-v2-keras#how-to-use)
5. [Original Weights](https://github.com/sebastian-sz/efficientnet-v2-keras#original-weights)

# Introduction
This is a package with EfficientNetV2 model variants adapted to Keras functional API.
I rewrote them this way so that the usage is similar to `keras.applications`.

The model's weights are converted from [original repository](https://github.com/google/automl/tree/master/efficientnetv2).

# Quickstart
You can use these models, similar to `keras.applications`:

```python
# Install
!pip install git+https://github.com/sebastian-sz/efficientnet-v2-keras@main

# Import package:
from efficientnet_v2 import EfficientNetV2S
import tensorflow as tf

# Use model directly:
model = EfficientNetV2S(
    weights='imagenet', input_shape=(384, 384, 3)
) 
model.summary()

# Or to extract features / fine tune:
backbone = EfficientNetV2S(
   weights='imagenet', 
   input_shape=(384, 384, 3),
   include_top=False
)

model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)  # 10 = num classes
])
model.compile(...)
model.fit(...)
```

You can fine tune these models, just like other Keras models.  

For end-to-end fine-tuning and conversion examples check out the 
[Colab Notebook](https://colab.research.google.com/drive/1CPTho02wBl48oOMqR2Wkj0xd90F3I9Uj?usp=sharing).

# Installation
There are multiple ways to install.  
The only requirements are Tensorflow 2.2+ and Python 3.6+.  
(Though, it is recommended to use **at least** Tensorflow 2.4)

### Option A: (recommended) pip install from github
`pip install git+https://github.com/sebastian-sz/efficientnet-v2-keras@main`

### Option B: Build from source
```bash
git clone https://github.com/sebastian-sz/efficientnet-v2-keras.git  
cd efficientnet-v2-keras  
pip install .
```

### Option C: (alternatively) no install:
If you do not want to install you could just drop the `efficientnet_v2/` directory, directly into your project.

### Option D: Docker
You can also install this package as an extension to official Tensorflow docker container:
Build: `docker build -t efficientnet_v2_keras .`  
Run: `docker run -it --rm efficientnet_v2_keras`

For GPU support or different TAG you can (for example) pass  
`--build-arg IMAGE_TAG=2.5.0-gpu`  
in build command.

### Verify installation
If all goes well you should be able to import:  
`from efficientnet_v2 import *`

# How to use

### Pretrained weights
Weights converted from original repository will be automatically downloaded, once you 
pass `weights="imagenet"` (or `imagenet-21k`, `imagenet-21k-ft1k`) upon model creation.

There are 3 weight variants: 
* `imagenet` - pretrained on Imagenet1k
* `imagenet-21k` - pretrained on Imagenet21k
* `imagenet-21k-ft1k` - pretrained on Imagenet21k and fine tuned on Imagenet1k

Note: `imagenet` weights have not been released for `XL` variant.

### Input shapes 
The variants expect the following input shapes.

| Model variant | Input shape |
|:-------------:|:-----------:|
|       B0      | `224,224`   |
|       B1      | `240,240`   |
|       B2      | `260,260`   |
|       B3      | `300,300`   |
|       S       | `384,384`   |
|       M       | `480,480`   |
|       L       | `480,480`   |
|       XL      | `512,512`   |

### Preprocessing
##### Option A: preprocessing function
The preprocessing is different for `Bx` and `S/M/L/XL` variants.
`Bx`'s expect image normalized with Imagenet mean and stddev, while other's a simple 
rescale:
```python
import tensorflow as tf

# Bx preprocessing:
def preprocess(image):  # input image is in range 0-255.
    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image -= tf.constant(mean_rgb, shape=(1, 1, 3), dtype=image.dtype)
    image /= tf.constant(stddev_rgb, shape=(1, 1, 3), dtype=image.dtype)
    return image
    
# S/M/L/XL preprocessing
def preprocess(image):  
    return (tf.cast(image, dtype=tf.float32) - 128.00) / 128.00
```
##### Option B: Preprocessing layers
or you can use [Preprocessing Layer](https://www.tensorflow.org/guide/keras/preprocessing_layers)
included in this repo:
```python
from efficientnet_v2 import get_preprocessing_layer

preprocessing_layer = get_preprocessing_layer(variant="b0")
```

### Fine-tuning
For fine-tuning example, check out the [Colab Notebook](https://colab.research.google.com/drive/1CPTho02wBl48oOMqR2Wkj0xd90F3I9Uj?usp=sharing).

### Tensorflow Lite
The models are TFLite compatible. You can convert them like any other Keras model:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("efficientnet_lite.tflite", "wb") as file:
  file.write(tflite_model)
```

### ONNX
The models are ONNX compatible. For ONNX Conversion you can use 
[tf2onnx](https://github.com/onnx/tensorflow-onnx) package:
```python
!pip install tf2onnx==1.8.4

# Save the model in TF's Saved Model format:
model.save("my_saved_model/")

# Convert:
!python -m tf2onnx.convert \
  --saved-model my_saved_model/ \
  --output efficientnet_v2.onnx
```
# Original Weights
The original weights are present in the
[original repoistory](https://github.com/google/automl/tree/master/efficientnetv2).
The original models were also trained using Keras are compatible with TF 2.

### (Optionally) Convert the weights
The converted weights are on this repository's GitHub. If, for some reason, you wish to 
download and convert original weights yourself, I prepared the utility scripts: 
1. `bash scripts/download_all.sh`
2. `bash scripts/convert_all.sh`

# Bibliography
[1] [Original repository](https://github.com/google/automl/tree/master/efficientnetv2)

# Closing words
If you found this repo useful, please consider giving it a star!

