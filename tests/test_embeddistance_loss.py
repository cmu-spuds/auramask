import unittest
import tensorflow as tf
from keras.metrics import CosineSimilarity

from auramask.losses.embeddistance import cosine_distance


class TestCosineDistance(unittest.TestCase):
    def setUp(self) -> None:
        self._image_shape = (128,)
        self._batch_image_shape = (5, 128)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._batch_image_shape
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed(self):
        a = tf.random.uniform(
            self._image_shape, minval=0, maxval=1.0, dtype=tf.float32, seed=123
        )
        b = tf.identity(a)
        dist = cosine_distance(a, b, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    # Test Opposite CD: 1
    def test_opposite_embed(self):
        a = tf.zeros(self._image_shape)
        b = tf.ones(self._image_shape)
        dist = cosine_distance(a, b, axis=-1)
        tf.debugging.assert_near(dist, 1.0)

    # Test Against Implementation:
    def test_against_tf(self):
        a = tf.ones(self._image_shape)
        b = tf.random.uniform(self._image_shape)
        dist = cosine_distance(a, b, axis=-1)
        tf_dist = 1 - CosineSimilarity()(a, b)
        tf.debugging.assert_near(dist, tf_dist)


if __name__ == "__main__":
    unittest.main()
