import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from test_efficientnet_v2 import utils
from test_efficientnet_v2.test_model import TEST_PARAMS

# Some conversions are RAM hungry and will crash CI on smaller machines. We skip those
# tests, not to break entire CI job.
MODEL_TO_MIN_MEMORY = {  # in GB
    "b0": 3.5,
    "b1": 4,
    "b2": 4,
    "b3": 4.5,
    "s": 5,
    "m": 7,
    "l": 11,
    "xl": 19,
}


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable, input_shape: Tuple[int, int]):
        # Skip test if not enough RAM:
        model_variant = self._testMethodName.split("_")[-1]
        minimum_required_ram = MODEL_TO_MIN_MEMORY[model_variant]
        if not utils.is_enough_memory(minimum_required_ram):
            self.skipTest(
                "Not enough memory to perform this test. Need at least "
                f"{minimum_required_ram} GB. Skipping... ."
            )

        model = model_fn(weights=None, input_shape=(*input_shape, 3))
        self._convert_and_save_tflite(model, input_shape)

        # Verify outputs:
        dummy_inputs = self.rng.uniform(shape=(1, *input_shape, 3))
        tflite_output = self._run_tflite_inference(dummy_inputs)
        self.assertTrue(isinstance(tflite_output, np.ndarray))
        self.assertEqual(tflite_output.shape, (1, 1000))

    def _convert_and_save_tflite(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        inference_func = utils.get_inference_function(model, input_shape)
        concrete_func = inference_func.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        tflite_model = converter.convert()
        with open(self.tflite_path, "wb") as file:
            file.write(tflite_model)

    def _run_tflite_inference(self, inputs: tf.Tensor) -> np.ndarray:
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], inputs.numpy())
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]["index"])


if __name__ == "__main__":
    absltest.main()
