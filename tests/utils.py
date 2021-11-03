from typing import Tuple

import tensorflow as tf


def get_inference_function(model: tf.keras.Model, input_shape: Tuple[int, int]):
    """Return convertible inference function with provided model."""

    def inference_func(inputs):
        return model(inputs, training=False)

    tensor_spec = tf.TensorSpec(shape=(1, *input_shape, 3), dtype=tf.float32)
    return tf.function(func=inference_func, input_signature=[tensor_spec])


def tensorflow_version_lower_than(target_version: float):
    """Return true if currently installed tensorflow is lower than specified."""
    current_version = float(".".join(tf.__version__.split(".")[:-1]))
    return current_version < target_version
