from typing import Callable
from typing import Tuple

import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from tests.test_efficientnet_v2 import TEST_PARAMS


class TestEfficientNetV2QATWrap(parameterized.TestCase):
    @parameterized.named_parameters(TEST_PARAMS)
    def test_qat_wrapping_entire_model(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        self.skipTest(
            "The entire model cannot be wrapped in Quantization Aware Training."
            "tensorflow.python.keras.layers.merge.Multiply layer is not supported."
            "This test might succeed, once TF-MOT package will be updated."
        )
        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        tfmot.quantization.keras.quantize_model(model)


if __name__ == "__main__":
    absltest.main()