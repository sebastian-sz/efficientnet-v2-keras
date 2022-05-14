import tensorflow as tf

if tf.__version__ < "2.8":
    from tensorflow.keras.layers.experimental.preprocessing import (
        Normalization,
        Rescaling,
    )
else:
    from tensorflow.keras.layers import Normalization, Rescaling


def get_preprocessing_layer(variant: str):
    """Return preprocessing layer for the given model variant."""
    if variant in {"b0", "b1", "b2", "b3"}:
        return Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
            axis=3 if tf.keras.backend.image_data_format() == "channels_last" else 1,
        )
    elif variant in {"s", "m", "l", "xl"}:
        return Rescaling(scale=1.0 / 128.0, offset=-1)
    else:
        raise ValueError(
            "Got unsupported variant. Must be either b0, b1, b2, b3, "
            f"s, m, l, xl. Received: {variant}"
        )
