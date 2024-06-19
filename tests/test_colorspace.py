import unittest
from keras import ops, random

from auramask.utils.colorspace import ColorSpaceEnum
from numpy import testing


class ColorTransformMethods(unittest.TestCase):
    def setUp(self) -> None:
        self._test_img = ops.convert_to_tensor(
            random.uniform(
                (224, 224, 3), minval=0, maxval=1.0, dtype="float32", seed=123
            )
        )
        self._test_batch_imgs = ops.convert_to_tensor(
            random.uniform(
                (5, 224, 224, 3), minval=0, maxval=1.0, dtype="float32", seed=456
            )
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
        self.assertLessEqual(ops.max(to_), 1.0)
        self.assertGreaterEqual(ops.min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_img.shape)
        testing.assert_allclose(to_, self._test_img)

    def test_to_and_from_rgb(self):
        processed_ = ColorSpaceEnum.RGB.value[0](self._test_img)
        processed_ = ColorSpaceEnum.RGB.value[1](processed_)
        self.assertLessEqual(ops.max(processed_), 1.0)
        self.assertGreaterEqual(ops.min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_img.shape)
        testing.assert_allclose(processed_, self._test_img)

    def test_to_rgb_batch(self):
        to_ = ColorSpaceEnum.RGB.value[0](self._test_batch_imgs)
        self.assertLessEqual(ops.max(to_), 1.0)
        self.assertGreaterEqual(ops.min(to_), 0.0)
        self.assertEqual(to_.shape, self._test_batch_imgs.shape)
        testing.assert_allclose(to_, self._test_batch_imgs)

    def test_to_and_from_rgb_batch(self):
        processed_ = ColorSpaceEnum.RGB.value[0](self._test_batch_imgs)
        processed_ = ColorSpaceEnum.RGB.value[1](processed_)
        self.assertLessEqual(ops.max(processed_), 1.0)
        self.assertGreaterEqual(ops.min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_batch_imgs.shape)
        testing.assert_allclose(processed_, self._test_batch_imgs)

    # RGB -> YUV -> RGB
    def test_to_yuv(self):
        to_ = ColorSpaceEnum.YUV.value[0](self._test_img)
        from tensorflow import image

        tf_to_ = image.rgb_to_yuv(self._test_img)

        # Test Y is in [0, 1]
        self.assertLessEqual(ops.max(to_[:, :, 0]), 1.0)
        self.assertGreaterEqual(ops.min(to_[:, :, 0]), 0.0)

        # Test U is in [-0.5, 0.5]
        self.assertLessEqual(ops.max(to_[:, :, 1]), 0.5)
        self.assertGreaterEqual(ops.min(to_[:, :, 1]), -0.5)

        # Test V is in [-0.5, 0.5]
        self.assertLessEqual(ops.max(to_[:, :, 2]), 0.5)
        self.assertGreaterEqual(ops.min(to_[:, :, 2]), -0.5)

        self.assertEqual(to_.shape, self._test_img.shape)
        testing.assert_allclose(
            ops.mean(tf_to_),
            ops.mean(to_),
            atol=self.atol,
            rtol=self.rtol,
        )
        # testing.assert_allclose(tf_to_, tf.subtract(to_, [0.,0.5,0.5]))

    def test_to_and_from_yuv(self):
        processed_ = ColorSpaceEnum.YUV.value[0](self._test_img)
        processed_ = ColorSpaceEnum.YUV.value[1](processed_)
        self.assertLessEqual(ops.max(processed_), 1.0)
        self.assertGreaterEqual(ops.min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_img.shape)
        testing.assert_allclose(
            ops.mean(processed_),
            ops.mean(self._test_img),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_to_yuv_batch(self):
        to_ = ColorSpaceEnum.YUV.value[0](self._test_batch_imgs)
        from tensorflow import image

        tf_to_ = image.rgb_to_yuv(self._test_batch_imgs)

        # Test Y is in [0, 1]
        self.assertLessEqual(ops.max(to_[:, :, :, 0]), 1.0)
        self.assertGreaterEqual(ops.min(to_[:, :, :, 0]), 0.0)

        # Test U is in [-0.5, 0.5]
        self.assertLessEqual(ops.max(to_[:, :, :, 1]), 0.5)
        self.assertGreaterEqual(ops.min(to_[:, :, :, 1]), -0.5)

        # Test V is in [-0.5, 0.5]
        self.assertLessEqual(ops.max(to_[:, :, :, 2]), 0.5)
        self.assertGreaterEqual(ops.min(to_[:, :, :, 2]), -0.5)

        self.assertEqual(to_.shape, self._test_batch_imgs.shape)
        testing.assert_allclose(
            ops.mean(to_, axis=[1, 2, 3]),
            ops.mean(tf_to_, axis=[1, 2, 3]),
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_to_and_from_yuv_batch(self):
        processed_ = ColorSpaceEnum.YUV.value[0](self._test_batch_imgs)
        processed_ = ColorSpaceEnum.YUV.value[1](processed_)
        self.assertLessEqual(ops.max(processed_), 1.0)
        self.assertGreaterEqual(ops.min(processed_), 0.0)
        self.assertEqual(processed_.shape, self._test_batch_imgs.shape)
        testing.assert_allclose(
            ops.mean(processed_, axis=[1, 2, 3]),
            ops.mean(self._test_batch_imgs, axis=[1, 2, 3]),
            atol=self.atol,
            rtol=self.rtol,
        )


if __name__ == "__main__":
    unittest.main()
