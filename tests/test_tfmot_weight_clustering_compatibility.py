from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from tests.test_efficientnet_v2 import TEST_PARAMS


class TestWeightClusteringWrappers(parameterized.TestCase):
    centroid_initialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        "number_of_clusters": 3,
        "cluster_centroids_init": centroid_initialization.DENSITY_BASED,
    }

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_weight_clustering_wrap(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        # total_ram = virtual_memory().total / (1024.0 ** 3)
        # if total_ram < 9 and not self._testMethodName.splendswith("b"):
        #     self.skipTest("Not enough memory to test weight clustering.")

        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        tfmot.clustering.keras.cluster_weights(model, **self.clustering_params)

    def setUp(self):
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    absltest.main()
