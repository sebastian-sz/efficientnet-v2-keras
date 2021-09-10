EfficientNetV2 models rewriteen in Keras functional API.

### Changelog:
* 10 Sept. 2021 - Added XL model variant.
    * Changed layer naming convention.
    * Rexported weights.


# Table of contens
1. [Introduction](https://github.com/sebastian-sz/efficientnet-v2-keras#introduction)
2. [Quickstart](https://github.com/sebastian-sz/efficientnet-v2-keras#quickstart)
3. [Installation](https://github.com/sebastian-sz/efficientnet-v2-keras#installation)
4. [How to use](https://github.com/sebastian-sz/efficientnet-v2-keras#how-to-use)
5. [Original Weights](https://github.com/sebastian-sz/efficientnet-v2-keras#original-weights)

# Introduction
This is a package with EfficientNetV2 model variants adapted to Keras functional API.
I reworte them this way so that the usage is similar to `keras.applications`.

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

For end-to-end fine tuning and conversion examples check out the 
[Colab Notebook](https://colab.research.google.com/drive/1CPTho02wBl48oOMqR2Wkj0xd90F3I9Uj?usp=sharing).

# Installation
There are multiple ways to install.  
The only requirements are Tensorflow 2.2+ and Python 3.6+.

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
There are in total 7 model variants you can use:   

| Model variant | Input shape | Weight variants* |
|:-------------:|:-----------:|:----------------:|
|       B0      | `224,224`   |   `imagenet`     |
|       B1      | `240,240`   |   `imagenet`     |
|       B2      | `260,260`   |   `imagenet`     |
|       B3      | `300,300`   |   `imagenet`     |
|       S       | `384,384`   |`imagenet, imagenet++`|
|       M       | `480,480`   |`imagenet, imagenet++`|
|       L       | `480,480`   |`imagenet, imagenet++`|

*`imagenet` means pretrained on Imagenet-1k dataset. `imagenet++` means 
pretrained on Imagenet-21k and fine tuned on Imagenet-1k.

### Preprocessing
The models expect image values in range `-1:+1`. In more detail the preprocessing 
function (for pretrained models) looks as follows:  
```python
def preprocess(image):  # input image is in range 0-255.
    return (tf.cast(image, dtype=tf.float32) - 128.00) / 128.00
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
The models are ONNX compatible. For ONNX Conversion you can use `tf2onnx` package:
```python
!pip install tf2onnx~=1.8.4

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

