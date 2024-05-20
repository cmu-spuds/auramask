import unittest
import tensorflow as tf

from auramask.losses import zero_dce as zdce

tf.config.run_functions_eagerly(True)  # For debugging


class TestColorConstancyLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.ColorConstancyLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "ColorConstancyLoss")

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )
        y_pred = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (4, 1, 1, 1))

        # Check if the result is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result)).numpy())

    # Note: This is true for the implementation as described in the paper. Other implementations square the difference twice.
    def test_call_method_known_values(self):
        # Define a test case with known values for deterministic output
        y_true = tf.constant(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype=tf.float32,
        )

        y_pred = tf.constant(
            [[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]],
            dtype=tf.float32,
        )

        expected_loss = tf.constant([[[[2.4494898]]]], dtype=tf.float32)

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss

        tf.debugging.assert_near(result, expected_loss)


class TestExposureControlLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.ExposureControlLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "ExposureControlLoss")
        tf.debugging.assert_near(self.loss.mean_val, 0.6)
        self.assertEqual(self.loss.window_size, 16)

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )
        y_pred = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertIsInstance(result, tf.Tensor)
        expected_shape = (
            4,
            64 // self.loss.window_size,
            64 // self.loss.window_size,
            1,
        )
        self.assertEqual(result.shape, expected_shape)

        # Check if the result is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result)).numpy())

    def test_call_method_known_values(self):
        self.loss.window_size = 1

        # Define a test case with known values for deterministic output
        y_true = tf.constant(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype=tf.float32,
        )

        y_pred = tf.constant(
            [[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]],
            dtype=tf.float32,
        )

        expected_loss = tf.constant([[[[0.01]]]], dtype=tf.float32)  # (0.5 - 0.6)^2

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss
        tf.debugging.assert_near(result, expected_loss)

        self.loss.window_size = 16


class TestIlluminationSmoothnessLoss(unittest.TestCase):
    def setUp(self):
        self.loss = zdce.IlluminationSmoothnessLoss()

    def test_initialization(self):
        self.assertEqual(self.loss.name, "IlluminationSmoothnessLoss")

    def test_call_method(self):
        # Create dummy tensors for y_true and y_pred
        y_true = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )
        y_pred = tf.random.uniform(
            (4, 64, 64, 3), minval=0, maxval=255, dtype=tf.float32
        )

        # Compute the loss
        result = self.loss.call(y_true, y_pred)

        # Check the type and shape of the result
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, ())

        # Check if the result is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result)).numpy())

    def test_call_method_known_values(self):
        # Define a test case with known values for deterministic output
        y_true = tf.constant(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype=tf.float32,
        )

        y_pred = tf.constant(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]],
            dtype=tf.float32,
        )

        expected_loss = tf.constant(0.29, dtype=tf.float32)  # Manually calculated

        result = self.loss.call(y_true, y_pred)

        # Compare the result with the expected loss (allowing some tolerance for floating-point errors)
        tf.debugging.assert_near(result, expected_loss)


if __name__ == "__main__":
    unittest.main()
