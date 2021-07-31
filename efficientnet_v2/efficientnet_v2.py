"""Code for EfficientNetV2 models."""
import copy
import math

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from efficientnet_v2.blocks_args import BLOCKS_ARGS

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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args=None,
    model_name="efficientnetv2",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Todo: add docstring."""
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

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,  # TODO: handle imagenet++
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
                name=f"blocks_{b}/",
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

    # TODO: weights loading logic.

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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
    """Todo: add docstring."""
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
