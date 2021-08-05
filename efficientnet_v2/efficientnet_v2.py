"""Code for EfficientNetV2 models."""
import copy
import math
from typing import Any, Callable, Dict, List, Tuple, Union

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from efficientnet_v2.blocks_args import BLOCKS_ARGS

BASE_WEIGHTS_URL = (
    "https://github.com/sebastian-sz/efficientnet-v2-keras/releases/download/v1.0/"
)
WEIGHT_HASHES = {
    "efficientnetv2-b0.h5": "c6b770b1c8cf213eb1399e9fbedf1871",
    "efficientnetv2-b1.h5": "79059d1067a7779887d3859706ef8480",
    "efficientnetv2-b2.h5": "b6c5c911b3cd7c8863d2aeb55b8ee1ee",
    "efficientnetv2-b3.h5": "e6bc1b2f04140a8eb1bf03d66343ea3a",
    "efficientnetv2-s.h5": "f0b49bdc045de8889f35234618edb59f",
    "efficientnetv2-m.h5": "9fb1ef92f80797b31fee575d1c0a24fe",
    "efficientnetv2-l.h5": "1e5d90cc5102212ba38cd7194c8d97d7",
    "efficientnetv2-b0_notop.h5": "8648ed1dd0b260705d02d29f8c651e91",
    "efficientnetv2-b1_notop.h5": "b859b006bc3fdbcad68be88c757d1b0a",
    "efficientnetv2-b2_notop.h5": "869924ed4837062b6a75f241b87c5afc",
    "efficientnetv2-b3_notop.h5": "090dd36d2024381bbbad4f8e4edcc30e",
    "efficientnetv2-s_notop.h5": "36cd089046169b7a1a2b3654ec2fa2a8",
    "efficientnetv2-m_notop.h5": "87a2dcf21014c367218c8495197fb35c",
    "efficientnetv2-l_notop.h5": "71f80290f1ae93e71c9ddd11e05ba721",
    "efficientnetv2-s-21k-ft1k.h5": "73d4916795840bb6cc3f1cd109e6858c",
    "efficientnetv2-m-21k-ft1k.h5": "7e4671a02dfe2673902f48c371bdbfd1",
    "efficientnetv2-l-21k-ft1k.h5": "2ad5eaaf1d1a48b3d7b544f306eaca51",
    "efficientnetv2-s-21k-ft1k_notop.h5": "534a11a6a4517d67b4d6dc021e642716",
    "efficientnetv2-m-21k-ft1k_notop.h5": "805410db76a6c7ada3202c4b61c40fc4",
    "efficientnetv2-l-21k-ft1k_notop.h5": "7a1233fdfe370c2a2e33a1b0af33f000",
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
        (pre-training on ImageNet), 'imagenet++ (pretrained on Imagenet 21k and fine
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

    if not (
        weights in {"imagenet", "imagenet++", None} or file_io.file_exists_v2(weights)
    ):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "`imagenet++` (ImageNet21K pretrained and finetuned) "
            "for s/m/l model variants,"
            "or the path to the weights file to be loaded."
        )

    if weights == ("imagenet" or "imagenet++") and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` or `"imagenet++"` with `include_top`'
            " as true, `classes` should be 1000"
        )
    if weights == "imagenet++" and model_name.split("-")[-1] not in {"s", "m", "l"}:
        raise ValueError(
            "Weights pretrained on 21k and fine tuned on 1k are only"
            "available for s-m-l model variants."
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights="imagenet" if weights in {"imagenet", "imagenet++"} else weights,
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
                name="block{}{}_".format(i + 1, chr(j + 97)),
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
    if weights == "imagenet" or weights == "imagenet++":
        weights_name = model_name

        if weights.endswith("++"):
            weights_name += "-21k-ft1k"

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
