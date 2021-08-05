import os
import tempfile
from typing import Callable, Tuple

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
)

TEST_PARAMS = [
    {"testcase_name": "b0", "model_fn": EfficientNetV2B0, "input_shape": (224, 224)},
    {"testcase_name": "b1", "model_fn": EfficientNetV2B1, "input_shape": (240, 240)},
    {"testcase_name": "b2", "model_fn": EfficientNetV2B2, "input_shape": (260, 260)},
    {"testcase_name": "b3", "model_fn": EfficientNetV2B3, "input_shape": (300, 300)},
    {"testcase_name": "s", "model_fn": EfficientNetV2S, "input_shape": (384, 384)},
    {"testcase_name": "m", "model_fn": EfficientNetV2M, "input_shape": (480, 480)},
    {"testcase_name": "l", "model_fn": EfficientNetV2L, "input_shape": (480, 480)},
]


class TestEfficientNetV2Unit(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    model_path = os.path.join(tempfile.mkdtemp(), "model.h5")

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_inference(self, model_fn: Callable, input_shape: Tuple[int, int]):
        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        mock_inputs = self.rng.uniform(shape=(1, *input_shape, 3), dtype=tf.float32)
        outputs = model(mock_inputs, training=False)

        self.assertTrue(isinstance(outputs, tf.Tensor))
        self.assertEqual(outputs.shape, (1, 1000))

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_keras_serialization(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        tf.keras.models.save_model(
            model=model, filepath=self.model_path, save_format="h5"
        )

        self.assertTrue(os.path.exists(self.model_path))

        loaded = tf.keras.models.load_model(self.model_path)
        self.assertTrue(isinstance(loaded, tf.keras.Model))

    def tearDown(self) -> None:
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


if __name__ == "__main__":
    absltest.main()
