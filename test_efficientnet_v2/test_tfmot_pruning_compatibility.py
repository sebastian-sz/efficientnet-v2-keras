from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_efficientnet_v2.test_model import TEST_PARAMS


class TestEfficientNetV2PruningWrapper(parameterized.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_pruning_entire_model(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        tfmot.sparsity.keras.prune_low_magnitude(model)


if __name__ == "__main__":
    absltest.main()
