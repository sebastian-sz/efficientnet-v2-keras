"""Script to convert original EfficientNet V2 weights to .h5 files."""
import json
import logging
from typing import Dict, List

import tensorflow as tf
from absl import app, flags

from efficientnet_v2.efficientnet_v2 import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model",
    default="",
    help="Model variant to convert. Must be one of: b0, b1, b2, b3, s, m, l.",
)
flags.DEFINE_string("ckpt", help="Path to .ckpt directory.", default="")
flags.DEFINE_string(
    "output", help="Path to converted .h5 file.", default="converted.h5"
)
flags.DEFINE_bool(
    "include_top",
    default=True,
    help="Whether to convert model with include_top option.",
)
flags.DEFINE_bool(
    "use_ema",
    default=False,
    help="Whether to use Exponential Moving Average variables during conversion.",
)


def main(argv):
    """Run conversion for target model variant and checkpoint."""
    arg_to_model_and_shape = {
        "b0": (EfficientNetV2B0, (224, 224)),
        "b1": (EfficientNetV2B1, (240, 240)),
        "b2": (EfficientNetV2B2, (260, 260)),
        "b3": (EfficientNetV2B3, (300, 300)),
        "s": (EfficientNetV2S, (384, 384)),
        "m": (EfficientNetV2M, (480, 480)),
        "l": (EfficientNetV2L, (480, 480)),
    }

    model_fn, input_shape = arg_to_model_and_shape[FLAGS.model]
    keras_model = model_fn(
        weights=None, input_shape=(*input_shape, 3), include_top=FLAGS.include_top
    )

    # Create mapping from keras to tensorflow ckpt blocks:
    logging.info("Mapping variables from ckpt to keras model...")

    keras_block_names = _get_keras_block_names(keras_model.variables)
    variable_names = get_variable_names_from_ckpt(FLAGS.ckpt, use_ema=FLAGS.use_ema)
    tf_block_names = _get_tf_block_names(variable_names)
    tf_to_keras_block_name_map = _get_keras_to_tf_block_map(
        keras_block_names=keras_block_names, tf_block_names=tf_block_names
    )

    tf_name_to_keras_map = _make_variable_names_map(
        variable_names=variable_names,
        tf_to_keras_block_name_map=tf_to_keras_block_name_map,
        use_ema=FLAGS.use_ema,
    )
    keras_name_to_tf_map = {v: k for k, v in tf_name_to_keras_map.items()}

    # Assign checkpoint variables to layer weights:
    logging.info("Converting weights...")
    progbar = tf.keras.utils.Progbar(target=len(keras_model.variables))

    for idx, variable in enumerate(keras_model.variables):
        original_variable_name = keras_name_to_tf_map.pop(variable.name)
        original_variable = tf.train.load_variable(FLAGS.ckpt, original_variable_name)
        variable.assign(original_variable)
        progbar.update(current=idx + 1)

    if list(keras_name_to_tf_map.keys()):
        logging.warning(
            f"Unused variables by this conversion: "
            f"{json.dumps(keras_name_to_tf_map, indent=4)}"
        )

    keras_model.save_weights(FLAGS.output, save_format="h5")
    logging.info(f"Done. Weights saved at {FLAGS.output}")


def _get_keras_to_tf_block_map(
    keras_block_names: List[str], tf_block_names: List[str]
) -> Dict[str, str]:
    assert len(tf_block_names) == len(keras_block_names), (
        f"Mismatched number of blocks. Found {len(tf_block_names)} in ckpt and "
        f"{len(keras_block_names)} in the model. Are you sure the ckpt is correct?"
    )
    return {
        tf_name: keras_name
        for tf_name, keras_name in zip(tf_block_names, keras_block_names)
    }


def _get_keras_block_names(keras_variables):
    results = []
    for variable in keras_variables:
        name = variable.name.split("/")[0].split("_")[0]  # block1a, block2a...
        if name.startswith("block") and name not in results:
            results.append(name)
    return sorted(results)


def _get_tf_block_names(tf_variable_names):
    results = []
    for variable_name in tf_variable_names:
        name = variable_name.split("/")[1]  # blocks_1, blocks_2...
        if name.startswith("blocks") and name not in results:
            results.append(name)
    return sorted(results, key=lambda x: int(x.split("_")[1]))


def get_variable_names_from_ckpt(path_ckpt, use_ema=False):
    """
    Get list of tensor names from checkpoint.

    Args:
        path_ckpt: str, path to the ckpt files
        use_ema: Bool, whether to use ExponentialMovingAverage result or not.

    Returns:
        List of variable names from checkpoint.
    """
    variables = tf.train.list_variables(path_ckpt)
    variable_names = [name for name, _ in variables if not name == "global_step"]

    if use_ema:
        variable_names = [x for x in variable_names if "ExponentialMovingAverage" in x]
    else:
        variable_names = [
            x for x in variable_names if "ExponentialMovingAverage" not in x
        ]

    return variable_names


def _make_variable_names_map(
    variable_names: List[str],
    tf_to_keras_block_name_map: Dict[str, str],
    use_ema: bool = False,
) -> Dict[str, str]:
    final_map = {}

    for var_name in variable_names:
        network_part = var_name.split("/")[1]  # head, stem, block0 ...
        if use_ema:
            last_name = var_name.split("/")[-2]  # kernel, dense....
        else:
            last_name = var_name.split("/")[-1]  # kernel, dense....

        # Handle stem:
        if network_part == "stem":
            stem_layer_map = _get_stem_map(var_name, last_name)
            final_map.update(stem_layer_map)

        # Handle head:
        if network_part == "head":
            head_layer_map = _get_head_map(var_name, last_name)
            final_map.update(head_layer_map)

        # Handle blocks:
        if network_part.startswith("blocks"):
            blocks_layer_map = _get_blocks_layer_map(
                var_name, last_name, tf_to_keras_block_name_map
            )
            final_map.update(blocks_layer_map)

    return final_map


def _get_blocks_layer_map(
    var_name: str, last_name: str, tf_to_keras_block_name_map: Dict[str, str]
):

    layer_type = var_name.split("/")[2]  # conv2d, tpu_batch_normalization...
    tf_block_name = var_name.split("/")[1]
    keras_block_name = tf_to_keras_block_name_map[tf_block_name]

    # 1st blocks only have project layer:
    if keras_block_name[5] == "1":
        if layer_type == "conv2d":
            keras_prefix = f"{keras_block_name}_project_conv/"
        elif layer_type == "tpu_batch_normalization":
            keras_prefix = f"{keras_block_name}_project_bn/"

    # Handle fused_mb_conv blocks:
    elif int(keras_block_name[5]) < 4:
        # expand conv:
        if layer_type == "conv2d":
            keras_prefix = f"{keras_block_name}_expand_conv/"
        # project conv:
        elif layer_type == "conv2d_1":
            keras_prefix = f"{keras_block_name}_project_conv/"
        # expand bn:
        elif layer_type == "tpu_batch_normalization":
            keras_prefix = f"{keras_block_name}_expand_bn/"
        # project bn:
        elif layer_type == "tpu_batch_normalization_1":
            keras_prefix = f"{keras_block_name}_project_bn/"

    # Handle mb conv blocks:
    else:
        if layer_type == "conv2d":
            keras_prefix = f"{keras_block_name}_expand_conv/"
        elif layer_type == "depthwise_conv2d":
            keras_prefix = f"{keras_block_name}_dwconv2/"
        elif layer_type == "se":
            layer_sub_type = var_name.split("/")[3]
            if layer_sub_type == "conv2d":
                keras_prefix = f"{keras_block_name}_se_reduce/"
            elif layer_sub_type == "conv2d_1":
                keras_prefix = f"{keras_block_name}_se_expand/"
        elif layer_type == "conv2d_1":
            keras_prefix = f"{keras_block_name}_project_conv/"

        elif layer_type == "tpu_batch_normalization":
            keras_prefix = f"{keras_block_name}_expand_bn/"
        elif layer_type == "tpu_batch_normalization_1":
            keras_prefix = f"{keras_block_name}_bn/"
        elif layer_type == "tpu_batch_normalization_2":
            keras_prefix = f"{keras_block_name}_project_bn/"

    return {var_name: f"{keras_prefix}{last_name}:0"}


def _get_head_map(var_name: str, last_name: str) -> Dict[str, str]:
    layer_type = var_name.split("/")[2]  # conv2d, tpu_batch_normalization...

    if layer_type == "conv2d":
        return {var_name: f"top_conv/{last_name}:0"}
    elif layer_type == "tpu_batch_normalization":
        return {var_name: f"top_bn/{last_name}:0"}
    elif layer_type == "dense":
        return {var_name: f"predictions/{last_name}:0"}
    else:
        raise ValueError(f"Unhandled variable: {var_name}")


def _get_stem_map(var_name: str, last_name: str) -> Dict[str, str]:
    layer_type = var_name.split("/")[2]  # conv2d, tpu_batch_normalization...

    if layer_type == "conv2d":
        return {var_name: f"stem_conv/{last_name}:0"}
    elif layer_type == "tpu_batch_normalization":
        return {var_name: f"stem_bn/{last_name}:0"}
    else:
        raise ValueError(f"Unhandled variable: {var_name}")


if __name__ == "__main__":
    app.run(main)
