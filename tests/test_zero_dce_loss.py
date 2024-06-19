import unittest
from keras import ops, random

from numpy import testing

from auramask.losses import zero_dce as zdce


class TestColorConstancyLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.ColorConstancyLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "ColorConstancyLoss")

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")
        y_pred = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertTrue(ops.is_tensor(result))
        self.assertEqual(result.shape, (4, 1, 1, 1))

        # Check if the result is finite
        self.assertTrue(ops.all(ops.isfinite(result)))

    # Note: This is true for the implementation as described in the paper. Other implementations square the difference twice.
    def test_call_method_known_values(self):
        # Define a test case with known values for deterministic output
        y_true = ops.convert_to_tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype="float32",
        )

        y_pred = ops.convert_to_tensor(
            [[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]],
            dtype="float32",
        )

        expected_loss = ops.convert_to_tensor([[[[2.4494898]]]], dtype="float32")

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss

        testing.assert_allclose(result, expected_loss)


class TestExposureControlLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.ExposureControlLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "ExposureControlLoss")
        testing.assert_almost_equal(self.loss.mean_val, 0.6)
        self.assertEqual(self.loss.window_size, 16)

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")
        y_pred = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertTrue(ops.is_tensor(result))
        expected_shape = (
            4,
            64 // self.loss.window_size,
            64 // self.loss.window_size,
            1,
        )
        self.assertEqual(result.shape, expected_shape)

        # Check if the result is finite
        self.assertTrue(ops.all(ops.isfinite(result)))

    def test_call_method_known_values(self):
        self.loss.window_size = 1

        # Define a test case with known values for deterministic output
        y_true = ops.convert_to_tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype="float32",
        )

        y_pred = ops.convert_to_tensor(
            [[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]],
            dtype="float32",
        )

        expected_loss = ops.convert_to_tensor(
            [[[[0.01]]]], dtype="float32"
        )  # (0.5 - 0.6)^2

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss
        testing.assert_allclose(result, expected_loss)

        self.loss.window_size = 16


class TestIlluminationSmoothnessLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.IlluminationSmoothnessLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "IlluminationSmoothnessLoss")

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")
        y_pred = random.uniform((4, 64, 64, 3), minval=0, maxval=255, dtype="float32")

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertTrue(ops.is_tensor(result))
        self.assertEqual(result.shape, ())

        # Check if the result is finite
        self.assertTrue(ops.all(ops.isfinite(result)))

    def test_call_method_known_values(self):
        # Define a test case with known values for deterministic output
        y_true = ops.convert_to_tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype="float32",
        )

        y_pred = ops.convert_to_tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]],
            dtype="float32",
        )

        expected_loss = ops.convert_to_tensor(
            0.29, dtype="float32"
        )  # Manually calculated

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss (allowing some tolerance for floating-point errors)
        testing.assert_allclose(result, expected_loss)


if __name__ == "__main__":
    unittest.main()
