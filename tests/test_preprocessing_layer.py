import tensorflow as tf
from absl.testing import absltest

from efficientnet_v2 import get_preprocessing_layer


class TestPreprocessingLayer(absltest.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()

    # TODO
    def setUp(self):
        if float(tf.__version__[:-2]) < 2.4:
            self.skipTest("Not supported for Tensorflow version below 2.4")

    def test_bx_variants_preprocessing_layer(self):
        def original_preprocess(image):
            mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            image -= tf.constant(mean_rgb, shape=(1, 1, 3), dtype=image.dtype)
            image /= tf.constant(stddev_rgb, shape=(1, 1, 3), dtype=image.dtype)
            return image

        input_frame = self.rng.uniform((1, 224, 224, 3), maxval=255)
        layer = get_preprocessing_layer("b0")

        original_preprocessed = original_preprocess(input_frame)
        layer_preprocessed = layer(input_frame)

        tf.debugging.assert_near(original_preprocessed, layer_preprocessed)

    def test_smlxl_variants_preprocessing_layer(self):
        def original_preprocess(image):
            return (image - 128.0) / 128.0

        input_frame = self.rng.uniform((1, 224, 224, 3), maxval=255)
        layer = get_preprocessing_layer("s")

        original_preprocessed = original_preprocess(input_frame)
        layer_preprocessed = layer(input_frame)

        tf.debugging.assert_near(original_preprocessed, layer_preprocessed)

    def test_get_preprocessing_layer_function(self):
        for variant in ["b0", "b1", "b2", "b3", "s", "m", "l", "xl"]:
            get_preprocessing_layer(variant)

        with self.assertRaises(ValueError):
            get_preprocessing_layer("non-existing-variant")


if __name__ == "__main__":
    absltest.main()
