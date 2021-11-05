"""Sanity check that the local converted weights are OK."""

import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from efficientnet_v2.efficientnet_v2 import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
    EfficientNetV2XL,
)
from root_dir import ROOT_DIR

OUTPUT_CONSISTENCY_TEST_PARAMS = [
    {
        "testcase_name": "b0",
        "model_fn": EfficientNetV2B0,
        "input_shape": (224, 224),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b0.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-b0_224_original_logits.npy",
        ),
    },
    {
        "testcase_name": "b1",
        "model_fn": EfficientNetV2B1,
        "input_shape": (240, 240),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b1.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-b1_240_original_logits.npy",
        ),
    },
    {
        "testcase_name": "b2",
        "model_fn": EfficientNetV2B2,
        "input_shape": (260, 260),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b2.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-b2_260_original_logits.npy",
        ),
    },
    {
        "testcase_name": "b3",
        "model_fn": EfficientNetV2B3,
        "input_shape": (300, 300),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b3.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-b3_300_original_logits.npy",
        ),
    },
    {
        "testcase_name": "s",
        "model_fn": EfficientNetV2S,
        "input_shape": (384, 384),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-s.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-s_384_original_logits.npy",
        ),
    },
    {
        "testcase_name": "m",
        "model_fn": EfficientNetV2M,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-m.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-m_480_original_logits.npy",
        ),
    },
    {
        "testcase_name": "l",
        "model_fn": EfficientNetV2L,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-l.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-l_480_original_logits.npy",
        ),
    },
    # 21k pretrained and 1k fine tuned variants:
    {
        "testcase_name": "s-21k-ft1k",
        "model_fn": EfficientNetV2S,
        "input_shape": (384, 384),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-s-21k-ft1k.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-s_384_original_logits_21k-ft1k.npy",
        ),
    },
    {
        "testcase_name": "m-21k-ft1k",
        "model_fn": EfficientNetV2M,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-m-21k-ft1k.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-m_480_original_logits_21k-ft1k.npy",
        ),
    },
    {
        "testcase_name": "l-21k-ft1k",
        "model_fn": EfficientNetV2L,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-l-21k-ft1k.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-l_480_original_logits_21k-ft1k.npy",
        ),
    },
    {
        "testcase_name": "xl-21k-ft1k",
        "model_fn": EfficientNetV2XL,
        "input_shape": (512, 512),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-xl-21k-ft1k.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_efficientnet_v2/assets/original_outputs/"
            "efficientnetv2-xl_512_original_logits_21k-ft1k.npy",
        ),
    },
]

FEATURE_EXTRACTION_TEST_PARAMS = [
    {
        "testcase_name": "b0-fe",
        "model_fn": EfficientNetV2B0,
        "input_shape": (224, 224),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b0_notop.h5"),
        "expected_feature_shape": (1, 7, 7, 1280),
    },
    {
        "testcase_name": "b1-fe",
        "model_fn": EfficientNetV2B1,
        "input_shape": (240, 240),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b1_notop.h5"),
        "expected_feature_shape": (1, 8, 8, 1280),
    },
    {
        "testcase_name": "b2-fe",
        "model_fn": EfficientNetV2B2,
        "input_shape": (260, 260),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b2_notop.h5"),
        "expected_feature_shape": (1, 9, 9, 1408),
    },
    {
        "testcase_name": "b3-fe",
        "model_fn": EfficientNetV2B3,
        "input_shape": (300, 300),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-b3_notop.h5"),
        "expected_feature_shape": (1, 10, 10, 1536),
    },
    {
        "testcase_name": "s-fe",
        "model_fn": EfficientNetV2S,
        "input_shape": (384, 384),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-s_notop.h5"),
        "expected_feature_shape": (1, 12, 12, 1280),
    },
    {
        "testcase_name": "m-fe",
        "model_fn": EfficientNetV2M,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-m_notop.h5"),
        "expected_feature_shape": (1, 15, 15, 1280),
    },
    {
        "testcase_name": "l-fe",
        "model_fn": EfficientNetV2L,
        "input_shape": (480, 480),
        "weights_path": os.path.join(ROOT_DIR, "weights/efficientnetv2-l_notop.h5"),
        "expected_feature_shape": (1, 15, 15, 1280),
    },
    {
        "testcase_name": "xl-fe",
        "model_fn": EfficientNetV2XL,
        "input_shape": (512, 512),
        "weights_path": os.path.join(
            ROOT_DIR, "weights/efficientnetv2-xl-21k-ft1k_notop.h5"
        ),
        "expected_feature_shape": (1, 16, 16, 1280),
    },
]


class TestLocalOutputConsistency(parameterized.TestCase):
    """Compare converted weight model outputs with original outputs."""

    image_path = os.path.join(ROOT_DIR, "test_efficientnet_v2/assets/panda.jpg")
    image = tf.image.decode_png(tf.io.read_file(image_path))

    @parameterized.named_parameters(OUTPUT_CONSISTENCY_TEST_PARAMS)
    def test_output_consistency(
        self,
        model_fn: Callable,
        input_shape: Tuple[int, int],
        weights_path: str,
        original_outputs: str,
    ):
        """Run model on sample image and compare output with original."""
        tf.keras.backend.clear_session()

        if not os.path.exists(weights_path):
            self.skipTest("No weights present locally. Skipping... .")

        model = model_fn(
            weights=None,
            input_shape=(*input_shape, 3),
            classifier_activation=None,  # Compare raw logits.
        )
        model.load_weights(weights_path)

        inputs = self._pre_process_image(self.image, input_shape=input_shape)
        outputs = model(inputs, training=False)

        original_outputs = np.load(original_outputs)

        tf.debugging.assert_near(outputs, original_outputs)

    @staticmethod
    def _pre_process_image(image: tf.Tensor, input_shape: Tuple[int, int]) -> tf.Tensor:
        """Preprocessing function from original repository.

        The official code can be found at
        https://github.com/google/automl/blob/c2ce63a63592d7b23cd023f5a519967029619fe2/efficientnetv2/preprocessing.py#L58
        The only difference is that I'm omitting the `image.set_shape` part.
        """
        image = (tf.cast(image, tf.float32) - 128.0) / 128.0
        transformations = "crop" if input_shape[0] < 320 else ""
        if "crop" in transformations:
            shape = tf.shape(image)
            height, width = shape[0], shape[1]
            ratio = input_shape[0] / (input_shape[0] + 32)  # for imagenet.
            crop_size = tf.cast(
                (ratio * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32
            )
            y, x = (height - crop_size) // 2, (width - crop_size) // 2
            image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
        image = tf.image.resize(image, input_shape)
        return tf.expand_dims(image, axis=0)

    @parameterized.named_parameters(FEATURE_EXTRACTION_TEST_PARAMS)
    def test_feature_extraction_with_pretrained_weights(
        self,
        model_fn: Callable,
        input_shape: Tuple[int, int],
        weights_path: str,
        expected_feature_shape: Tuple[int, int, int, int],
    ):
        """Check whether we can extract features via converted weights."""
        if not os.path.exists(weights_path):
            self.skipTest("No weights present locally. Skipping... .")

        model = model_fn(weights=None, input_shape=(*input_shape, 3), include_top=False)
        model.load_weights(weights_path)

        inputs = self._pre_process_image(self.image, input_shape=input_shape)

        features = model(inputs, training=False)

        self.assertTrue(isinstance(features, tf.Tensor))
        self.assertEqual(features.shape, expected_feature_shape)


if __name__ == "__main__":
    absltest.main()
