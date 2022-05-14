import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from test_efficientnet_v2.test_model import TEST_PARAMS
from test_efficientnet_v2.utils import get_inference_function

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

    _tolerance = 1e-5

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable, input_shape: Tuple[int, int]):
        # Skip test if not enough RAM:
        model_variant = self._testMethodName.split("_")[-1]
        if not self._enough_memory_to_convert(model_variant):
            self.skipTest(
                "Not enough memory to convert to tflite. Need at least "
                f"{MODEL_TO_MIN_MEMORY[model_variant]} GB. Skipping... ."
            )

        # Comparison will fail with random weights as we are comparing
        # very low floats.
        # Load XL variant with imagenet++ weights as these are only available.
        weights_arg = "imagenet-21k-ft1k" if input_shape == (512, 512) else "imagenet"
        model = model_fn(weights=weights_arg, input_shape=(*input_shape, 3))

        self._convert_and_save_tflite(model, input_shape)
        self.assertTrue(os.path.isfile(self.tflite_path))

        # Check outputs:
        mock_input = self.rng.uniform(shape=(1, *input_shape, 3))

        original_output = model(mock_input, training=False)
        tflite_output = self._run_tflite_inference(mock_input)

        tf.debugging.assert_near(
            original_output, tflite_output, rtol=self._tolerance, atol=self._tolerance
        )

    def _convert_and_save_tflite(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        inference_func = get_inference_function(model, input_shape)
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

    @staticmethod
    def _enough_memory_to_convert(model_name: str) -> bool:
        total_ram = virtual_memory().total / (1024.0**3)
        required_ram = MODEL_TO_MIN_MEMORY[model_name]
        return total_ram >= required_ram


if __name__ == "__main__":
    absltest.main()
