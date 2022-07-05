"""Code for EfficientNetV2 models."""
import copy
import math
import sys
from typing import Any, Callable, Dict, List, Tuple, Union

import tensorflow as tf
from absl import logging
from packaging import version
from tensorflow.python.keras import backend

# Keras has been moved to separate repository
if version.parse(tf.__version__) < version.parse("2.8"):
    from tensorflow.python.keras.applications import imagenet_utils
else:
    from keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from efficientnet_v2.blocks_args import BLOCKS_ARGS

BASE_WEIGHTS_URL = (
    "https://github.com/sebastian-sz/efficientnet-v2-keras/releases/download/v2.0/"
)

WEIGHT_HASHES = {
    # Imagenet 1k
    "efficientnetv2-b0.h5": "040bd13d0e1120f3d3ff64dcb1b311da",
    "efficientnetv2-b0_notop.h5": "0ee6a45fb049baaaf5dd710e50828382",
    "efficientnetv2-b1.h5": "2e640a47676a72aab97fbcd5cdc5aee5",
    "efficientnetv2-b1_notop.h5": "650f09a0e2d4282201b5187ac2709721",
    "efficientnetv2-b2.h5": "ff25e799dd33de560322a2f0bfba1b53",
    "efficientnetv2-b2_notop.h5": "4236cc709ddb4616c81c877b3f92457f",
    "efficientnetv2-b3.h5": "7a9f26b46c88c64a428ca998fa31e9d4",
    "efficientnetv2-b3_notop.h5": "cb807fb01931c554fd00ae79d5b9cf4d",
    "efficientnetv2-m.h5": "4766229c2bd41aa09c7271e3c3a5403d",
    "efficientnetv2-m_notop.h5": "4bb03763f7be9b3829a3e640c358de17",
    "efficientnetv2-s.h5": "6cb2135fe05dbd9ced79348b8b76f05f",
    "efficientnetv2-s_notop.h5": "551df41bf4f0951006926610e93c17c1",
    "efficientnetv2-l.h5": "25db7bfb451abc977bcc4140c91c4e9e",
    "efficientnetv2-l_notop.h5": "451021c40955e974b7627b9e588211a1",
    # Imagenet 21k
    "efficientnetv2-b0-21k.h5": "8635973271bb9a88eaee549ff54aedfe",
    "efficientnetv2-b0-21k_notop.h5": "3f28d90919518ef426073dbcb17e3021",
    "efficientnetv2-b1-21k.h5": "769d9b75be3438f1b6097235bde22028",
    "efficientnetv2-b1-21k_notop.h5": "611cfd8977562c93bc4959992ad9bd48",
    "efficientnetv2-b2-21k.h5": "d9398206a6d2859d3bf45f6f524caa08",
    "efficientnetv2-b2-21k_notop.h5": "7467240653f73dd438e87af589a859ad",
    "efficientnetv2-b3-21k.h5": "a162c5a30af3244445f6a633ae29f82c",
    "efficientnetv2-b3-21k_notop.h5": "d2629d05829af1450432e2f114ce2917",
    "efficientnetv2-s-21k.h5": "6629e2eb68b6ebc922e009f6f800ad51",
    "efficientnetv2-s-21k_notop.h5": "c8ddbae1744f089f630f2bdbad5fe2fa",
    "efficientnetv2-m-21k.h5": "996706525ce91d0113b2653099c64ec9",
    "efficientnetv2-m-21k_notop.h5": "7691b54d75412ca020aacfcb2a5837c6",
    "efficientnetv2-l-21k.h5": "43ae5d74761ce151bbc0fb552184e378",
    "efficientnetv2-l-21k_notop.h5": "7ce647fe4de717b57a5fd6f2b3c82843",
    "efficientnetv2-xl-21k.h5": "3b9760ecac79f6d0b0fe9648f14a2fed",
    "efficientnetv2-xl-21k_notop.h5": "456d3fdcfc95bb460fcad7f0d8095773",
    # Imagenet 21k-ft1k
    "efficientnetv2-b0-21k-ft1k.h5": "bc8fe2c555e5a1229c378d0e84aa2703",
    "efficientnetv2-b0-21k-ft1k_notop.h5": "9963bc6b7aa74eac7036ab414dff9733",
    "efficientnetv2-b1-21k-ft1k.h5": "872bddc747d40c6238c964fe73a3a1e6",
    "efficientnetv2-b1-21k-ft1k_notop.h5": "f600737b414724d659c2bb7b5465aa22",
    "efficientnetv2-b2-21k-ft1k.h5": "08fd7f48575c7a3a852c026f300e6a3f",
    "efficientnetv2-b2-21k-ft1k_notop.h5": "78c435611d5aa909e725f40a7a1119bf",
    "efficientnetv2-b3-21k-ft1k.h5": "c1a195289bb3574caac5f2c94cd7f011",
    "efficientnetv2-b3-21k-ft1k_notop.h5": "99f66b5aa597a8834ba74f0b5d8a81d7",
    "efficientnetv2-s-21k-ft1k.h5": "62a850f1b111c4872277c18d64b928d4",
    "efficientnetv2-s-21k-ft1k_notop.h5": "85d8dcc7a63523abea94469b833be01e",
    "efficientnetv2-m-21k-ft1k.h5": "8f6f7ca84d948da4b93f4b9053c19413",
    "efficientnetv2-m-21k-ft1k_notop.h5": "f670a1cb04aeed321c554c21f219f895",
    "efficientnetv2-l-21k-ft1k.h5": "78e5ffa224184f1481252a115a5f003d",
    "efficientnetv2-l-21k-ft1k_notop.h5": "5a4795a11ae52a7d8626c9e20ba275a5",
    "efficientnetv2-xl-21k-ft1k.h5": "f48b9f1c12effdf9d70a33d81eb9f5ca",
    "efficientnetv2-xl-21k-ft1k_notop.h5": "a0cbe206c87e8fafe7434451e5ac79a9",
}
CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


def mb_conv_block(
    inputs,
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name: str = "",
):
    """Create MBConv block: Mobile Inverted Residual Bottleneck."""
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # Expansion phase
    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=name + "expand_conv",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "expand_bn"
        )(x)
        x = tf.keras.layers.Activation(activation, name=name + "expand_activation")(x)
    else:
        x = inputs

    # Depthwise conv.
    # Do not use zero pad. Strides is literally always 1.
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "dwconv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "bn"
    )(x)
    x = tf.keras.layers.Activation(activation, name=name + "activation")(x)

    # Skip conv_dropout - is always null.

    # Squeeze and excite:
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        if bn_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = tf.keras.layers.Reshape(se_shape, name=name + "se_reshape")(se)

        se = tf.keras.layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_reduce",
        )(se)
        se = tf.keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_expand",
        )(se)

        x = tf.keras.layers.multiply([x, se], name=name + "se_excite")

    # Output phase
    x = tf.keras.layers.Conv2D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "project_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn"
    )(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:  # They made the same thing for efficientnet V1.
            x = tf.keras.layers.Dropout(
                survival_probability, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)

        x = tf.keras.layers.add([x, inputs], name=name + "add")
    return x


def fused_mb_conv_block(
    inputs,
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name: str = "",
):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format="channels_last",
            padding="same",
            use_bias=False,
            name=name + "expand_conv",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "expand_bn"
        )(x)
        x = tf.keras.layers.Activation(
            activation=activation, name=name + "expand_activation"
        )(x)
    else:
        x = inputs

    # Skip conv_dropout - is always null.

    # Squeeze and excite
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        if bn_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = tf.keras.layers.Reshape(se_shape, name=name + "se_reshape")(se)

        se = tf.keras.layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_reduce",
        )(se)
        se = tf.keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_expand",
        )(se)

        x = tf.keras.layers.multiply([x, se], name=name + "se_excite")

    # Output phase:
    x = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1 if expand_ratio != 1 else strides,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        name=name + "project_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn"
    )(x)
    if expand_ratio == 1:
        x = tf.keras.layers.Activation(
            activation=activation, name=name + "project_activation"
        )(x)

    # Residual:
    if strides == 1 and input_filters == output_filters:
        if survival_probability:  # They made the same thing for efficientnet V1.
            x = tf.keras.layers.Dropout(
                survival_probability, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)

        x = tf.keras.layers.add([x, inputs], name=name + "add")
    return x


def EfficientNetV2(
    width_coefficient: float,
    depth_coefficient: float,
    default_size: int,
    dropout_rate=0.2,
    drop_connect_rate: float = 0.2,
    depth_divisor: int = 8,
    min_depth: int = 8,
    bn_momentum: float = 0.9,
    activation="swish",
    blocks_args: List[Dict[str, Any]] = None,
    model_name: str = "efficientnetv2",
    include_top: bool = True,
    weights: str = "imagenet",
    input_tensor=None,
    input_shape: Tuple[int, int, int] = None,
    pooling=None,
    classes: int = 1000,
    classifier_activation: Union[str, Callable] = "softmax",
):
    """Instantiate the EfficientNetV2 architecture using given provided parameters.

    :param width_coefficient: scaling coefficient for network width.
    :param depth_coefficient: scaling coefficient for network depth.
    :param default_size: default input image size.
    :param dropout_rate: dropout rate before final classifier layer.
    :param drop_connect_rate: dropout rate at skip connections.
    :param depth_divisor: a unit of network width.
    :param min_depth: integer, minimum number of filters.
    :param bn_momentum: Momentum parameter for Batch Normalization layers.
    :param activation: activation function.
    :param blocks_args: list of dicts, parameters to construct block modules.
    :param model_name: name of the model.
    :param include_top: whether to include the fully-connected layer at the top of
        the network.
    :param weights: one of `None` (random initialization), 'imagenet'
        (pre-training on ImageNet), 'imagenet-21k' (pretrained on Imagenet21k),
        'imagenet21k-ft1k' (pretrained on Imagenet 21k and fine
        tuned on 1k)' or the path to the weights file to be loaded.
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
        as image input for the model.
    :param input_shape: optional shape tuple, only to be specified if `include_top` is
        False. It should have exactly 3 inputs channels.
    :param  pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    :param classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is specified.
    :param classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `tf.keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`, or invalid input
        shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    if not blocks_args:
        blocks_args = BLOCKS_ARGS[model_name]

    if weights == "imagenet++":
        weights = "imagenet-21k-ft1k"
        logging.warning(
            "imagenet++ argument is deprecated. "
            "Please use imagenet-21k-ft1k instead."
        )

    if not (
        weights in {"imagenet", "imagenet-21k-ft1k", "imagenet-21k", None}
        or file_io.file_exists_v2(weights)
    ):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "`imagenet-21k-ft1k` (ImageNet21K pretrained and finetuned on Imagenet 1k),"
            "`imagenet-21k` (pretrained on ImageNet21k) "
            "or the path to the weights file to be loaded."
            f"Received weights={weights}"
        )

    if (
        weights == ("imagenet" or "imagenet-21k-ft1k")
        and include_top
        and classes != 1000
    ):
        raise ValueError(
            f"If using `weights` as `'imagenet'` or `'imagenet-21k-ft1k'` with "
            f"`include_top` as true, `classes` should be 1000. "
            f"Received classes={classes}"
        )
    if weights == "imagenet-21k" and include_top and classes != 21843:
        raise ValueError(
            f"If using `weights` as `imagenet-21k` with `include_top` as"
            f"true, `classes` should be 21843. Received classes={classes}"
        )

    if model_name.split("-")[-1] == "xl" and weights == "imagenet":
        raise ValueError(
            "XL variant does not have `imagenet` weights released. Please use"
            "`imagenet-21k` or `imagenet-21k-ft1k`."
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights="imagenet"
        if weights in {"imagenet", "imagenet-21k", "imagenet-21k-ft1k"}
        else weights,
    )

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def round_filters(filters):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        minimum_depth = min_depth or depth_divisor
        new_filters = max(
            minimum_depth,
            int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
        )
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of filters based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input

    stem_filters = blocks_args[0]["input_filters"]
    x = tf.keras.layers.Conv2D(
        filters=round_filters(stem_filters),
        kernel_size=3,
        strides=2,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, epsilon=0.001, name="stem_bn"
    )(x)
    x = tf.keras.layers.Activation(activation, name="stem_activation")(x)

    # # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    for (i, args) in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        args["input_filters"] = round_filters(args["input_filters"])
        args["output_filters"] = round_filters(args["output_filters"])

        conv_block = {0: mb_conv_block, 1: fused_mb_conv_block}[args.pop("conv_type")]

        for j in range(round_repeats(args.pop("num_repeat"))):

            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = conv_block(
                inputs=x,
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name=f"block{i+1}-{j+1:02d}_",
                **args,
            )
            b += 1

    # Build top
    x = tf.keras.layers.Conv2D(
        filters=round_filters(1280),  # All feature sizes are the same.
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name="top_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name="top_bn"
    )(x)
    x = tf.keras.layers.Activation(activation=activation, name="top_activation")(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(0),
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Download weights:
    if weights in {"imagenet", "imagenet-21k", "imagenet-21k-ft1k"}:
        weights_name = model_name

        if weights.endswith("21k-ft1k"):
            weights_name += "-21k-ft1k"
        elif weights.endswith("21k"):
            weights_name += "-21k"

        if not include_top:
            weights_name += "_notop"

        filename = f"{weights_name}.h5"
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = tf.keras.utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir="models",
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetV2S(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 S variant."""
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=384,
        model_name="efficientnetv2-s",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2M(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 M variant."""
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-m",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2L(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 L variant."""
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-l",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2B0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 B0 variant."""
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        model_name="efficientnetv2-b0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2B1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 B1 variant."""
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        model_name="efficientnetv2-b1",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2B2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 B2 variant."""
    return EfficientNetV2(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        model_name="efficientnetv2-b2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2B3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 B3 variant."""
    return EfficientNetV2(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=300,
        model_name="efficientnetv2-b3",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetV2XL(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Create EfficientNetV2 XL variant."""
    # This model is so big that it's creation exceeds default recursion limit
    current_limit = sys.getrecursionlimit()
    target_limit = 2000
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)

    model = EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=512,
        model_name="efficientnetv2-xl",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )

    return model
