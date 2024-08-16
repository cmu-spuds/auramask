import unittest
from keras import metrics, random, ops, backend as K
from numpy import testing

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
        a = random.uniform(
            self._image_shape, minval=0, maxval=1.0, dtype=K.floatx(), seed=123
        )
        b = ops.copy(a)
        dist = cosine_distance(a, b, axis=-1)
        testing.assert_almost_equal(dist, 0.0)

    # Test Opposite CD: 1
    def test_opposite_embed(self):
        a = ops.zeros(self._image_shape)
        b = ops.ones(self._image_shape)
        dist = cosine_distance(a, b, axis=-1)
        testing.assert_almost_equal(dist, 1.0)

    # Test Against Implementation:
    def test_against_tf(self):
        a = ops.ones(self._image_shape)
        b = random.uniform(self._image_shape)
        dist = cosine_distance(a, b, axis=-1)
        tf_dist = 1 - metrics.CosineSimilarity()(a, b)
        testing.assert_almost_equal(dist, tf_dist)


if __name__ == "__main__":
    unittest.main()
