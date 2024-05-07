import unittest
import tensorflow as tf

from auramask.utils.colorspace import ColorSpaceEnum


class ColorTransformMethods(unittest.TestCase):
    def setUp(self) -> None:
        self._test_img = tf.random.uniform(
            (224, 224, 3), minval=0, maxval=1.0, dtype=tf.float32, seed=123
        )
        self._test_batch_imgs = tf.random.uniform(
            (5, 224, 224, 3), minval=0, maxval=1.0, dtype=tf.float32, seed=456
        )
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._test_img
        del self._test_batch_imgs
        return super().tearDownClass()

    # RGB -> RGB -> RGB
    def test_to_rgb(self):
        to_ = ColorSpaceEnum.RGB.value[0](self._test_img)
        self.assertLessEqual(tf.reduce_max(to_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_img.shape)
        tf.debugging.assert_near(to_, self._test_img)

    def test_to_and_from_rgb(self):
        processed_ = ColorSpaceEnum.RGB.value[0](self._test_img)
        processed_ = ColorSpaceEnum.RGB.value[1](processed_)
        self.assertLessEqual(tf.reduce_max(processed_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_img.shape)
        tf.debugging.assert_near(processed_, self._test_img)

    def test_to_rgb_batch(self):
        to_ = ColorSpaceEnum.RGB.value[0](self._test_batch_imgs)
        self.assertLessEqual(tf.reduce_max(to_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_batch_imgs.shape)
        tf.debugging.assert_near(to_, self._test_batch_imgs)

    def test_to_and_from_rgb_batch(self):
        processed_ = ColorSpaceEnum.RGB.value[0](self._test_batch_imgs)
        processed_ = ColorSpaceEnum.RGB.value[1](processed_)
        self.assertLessEqual(tf.reduce_max(processed_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_batch_imgs.shape)
        tf.debugging.assert_near(processed_, self._test_batch_imgs)

    # RGB -> YUV -> RGB
    def test_to_yuv(self):
        to_ = ColorSpaceEnum.YUV.value[0](self._test_img)
        shifted_to_ = tf.subtract(to_, [0.0, 0.5, 0.5])
        tf_to_ = tf.image.rgb_to_yuv(self._test_img)

        # Test Y is in [0, 1]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, 0]), 1.0)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, 0]), 0.0)

        # Test U is in [-0.5, 0.5]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, 1]), 0.5)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, 1]), -0.5)

        # Test V is in [-0.5, 0.5]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, 2]), 0.5)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, 2]), -0.5)

        self.assertLessEqual(tf.reduce_max(to_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_img.shape)
        tf.debugging.assert_near(
            tf.reduce_mean(tf_to_),
            tf.reduce_mean(shifted_to_),
            atol=self.atol,
            rtol=self.rtol,
        )
        # tf.debugging.assert_near(tf_to_, tf.subtract(to_, [0.,0.5,0.5]))

    def test_to_and_from_yuv(self):
        processed_ = ColorSpaceEnum.YUV.value[0](self._test_img)
        processed_ = ColorSpaceEnum.YUV.value[1](processed_)
        self.assertLessEqual(tf.reduce_max(processed_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_img.shape)
        tf.debugging.assert_near(
            tf.reduce_mean(processed_),
            tf.reduce_mean(self._test_img),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_to_yuv_batch(self):
        to_ = ColorSpaceEnum.YUV.value[0](self._test_batch_imgs)
        tf_to_ = tf.image.rgb_to_yuv(self._test_batch_imgs)
        shifted_to_ = tf.subtract(to_, [0.0, 0.5, 0.5])
        self.assertLessEqual(tf.reduce_max(to_), 1.0)

        # Test Y is in [0, 1]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, :, 0]), 1.0)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, :, 0]), 0.0)

        # Test U is in [-0.5, 0.5]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, :, 1]), 0.5)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, :, 1]), -0.5)

        # Test V is in [-0.5, 0.5]
        tf.debugging.assert_less_equal(tf.reduce_max(shifted_to_[:, :, :, 2]), 0.5)
        tf.debugging.assert_greater_equal(tf.reduce_min(shifted_to_[:, :, :, 2]), -0.5)

        self.assertGreaterEqual(tf.reduce_min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_batch_imgs.shape)
        tf.debugging.assert_near(
            tf.reduce_mean(shifted_to_, axis=[1, 2, 3]),
            tf.reduce_mean(tf_to_, axis=[1, 2, 3]),
            summarize=5,
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_to_and_from_yuv_batch(self):
        processed_ = ColorSpaceEnum.YUV.value[0](self._test_batch_imgs)
        processed_ = ColorSpaceEnum.YUV.value[1](processed_)
        self.assertLessEqual(tf.reduce_max(processed_), 1.0)
        self.assertGreaterEqual(tf.reduce_min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_batch_imgs.shape)
        tf.debugging.assert_near(
            tf.reduce_mean(processed_, axis=[1, 2, 3]),
            tf.reduce_mean(self._test_batch_imgs, axis=[1, 2, 3]),
            summarize=5,
            atol=self.atol,
            rtol=self.rtol,
        )


if __name__ == "__main__":
    unittest.main()
