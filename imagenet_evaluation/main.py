import tensorflow as tf
from absl import app, flags
from external import datasets

import efficientnet_v2

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir",
    default="/workspace/tfrecords/validation/",
    help="Path to validation tfrecords directory (generaed by `imagenet_to_gcs.py`).",
)
flags.DEFINE_enum(
    "variant", "b0", ["b0", "b1", "b2", "b3", "s", "m", "l"], "Model variant to use."
)
flags.DEFINE_integer("batch_size", default=16, help="Batch size for eval.")
flags.DEFINE_string(
    "weights", default="imagenet", help="Path to weights or weights argument"
)
flags.DEFINE_bool(
    "use_tfhub",
    default=False,
    help="Whether to evaluate TFHub models instead of this repo.",
)

VARIANT_TO_KERAS = {
    "b0": efficientnet_v2.EfficientNetV2B0,
    "b1": efficientnet_v2.EfficientNetV2B1,
    "b2": efficientnet_v2.EfficientNetV2B2,
    "b3": efficientnet_v2.EfficientNetV2B3,
    "s": efficientnet_v2.EfficientNetV2S,
    "m": efficientnet_v2.EfficientNetV2M,
    "l": efficientnet_v2.EfficientNetV2L,
}

HUB_BASE_URL = "https://tfhub.dev/google/imagenet/"
VARIANT_TO_HUB_URL = {
    "b0": "efficientnet_v2_imagenet1k_b0/classification/2",
    "b1": "efficientnet_v2_imagenet1k_b1/classification/2",
    "b2": "efficientnet_v2_imagenet1k_b2/classification/2",
    "b3": "efficientnet_v2_imagenet1k_b3/classification/2",
    "s": "efficientnet_v2_imagenet1k_s/classification/2",
    "m": "efficientnet_v2_imagenet1k_m/classification/2",
    "l": "efficientnet_v2_imagenet1k_l/classification/2",
}


VARIANT_TO_SIZE = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "s": 384,
    "m": 480,
    "l": 480,
}


def _strip_dict(images, labels):
    return images["image"], labels["label"]


def main(argv_):
    """Run Imagenet eval job."""
    # Load models:
    image_size = VARIANT_TO_SIZE[FLAGS.variant]
    if FLAGS.use_tfhub:
        import tensorflow_hub as tf_hub  # Local import so that this is optional.

        model_url = HUB_BASE_URL + VARIANT_TO_HUB_URL[FLAGS.variant]
        model = tf_hub.KerasLayer(model_url)
        model.build([None, image_size, image_size, 3])
    else:
        model = VARIANT_TO_KERAS[FLAGS.variant](weights=FLAGS.weights)

    # Load data
    ds_class = datasets.get_dataset_class("imagenet")
    val_dataset = ds_class(
        is_training=False,
        data_dir=FLAGS.data_dir,
        cache=False,
        image_size=image_size,
        image_dtype=None,
        augname="effnetv1_autoaug" if FLAGS.variant.startswith("b") else None,
        mixup_alpha=0,
        ra_num_layers=2,
        ra_magnitude=20,
    )
    params = {"batch_size": FLAGS.batch_size}
    val_dataset = val_dataset.input_fn(params).map(
        _strip_dict, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Run eval:
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    progbar = tf.keras.utils.Progbar(target=50000 // FLAGS.batch_size)

    for idx, (images, y_true) in enumerate(val_dataset):
        y_pred = model(images, training=False)

        top1.update_state(y_true=y_true, y_pred=y_pred)
        top5.update_state(y_true=y_true, y_pred=y_pred)

        progbar.update(
            idx, [("top1", top1.result().numpy()), ("top5", top5.result().numpy())]
        )

    print()
    print(f"TOP1: {top1.result().numpy()}.  TOP5: {top5.result().numpy()}")


if __name__ == "__main__":
    app.run(main)
