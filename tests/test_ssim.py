# ruff: noqa: E402

import os

os.environ["KERAS_BACKEND"] = "torch"

import unittest
from keras import ops, random, backend as K, config

from numpy import testing
from skimage.metrics import structural_similarity as compare_ssim
from auramask.losses.ssim import DSSIMObjective

config.disable_traceback_filtering()
K.set_image_data_format("channels_last")


class TestSSIM(unittest.TestCase):
    def setUp(self) -> None:
        self._test_img_a = ops.convert_to_tensor(
            random.uniform(
                (224, 224, 3), minval=0, maxval=1.0, dtype=K.floatx(), seed=123
            )
        )
        self.ssim = DSSIMObjective()
        self.decimal = 3
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._test_img_a
        return super().tearDownClass()

    # SSIM same
    def test_ssim_same(self):
        t_ssim = (
            1
            - compare_ssim(
                ops.convert_to_numpy(self._test_img_a),
                ops.convert_to_numpy(self._test_img_a),
                data_range=1.0,
                channel_axis=2,
                window_size=3,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ) / 2
        p_ssim = self.ssim(self._test_img_a, self._test_img_a)
        testing.assert_almost_equal(
            t_ssim, ops.convert_to_numpy(p_ssim), decimal=self.decimal
        )

    # SSIM same with small noise
    def test_ssim_noise(self):
        _test_img_b = ops.clip(
            ops.add(
                self._test_img_a,
                ops.convert_to_tensor(
                    random.uniform(
                        (224, 224, 3),
                        minval=-0.1,
                        maxval=0.1,
                        dtype=K.floatx(),
                        seed=456,
                    )
                ),
            ),
            0.0,
            1.0,
        )
        t_ssim = (
            1
            - compare_ssim(
                ops.convert_to_numpy(self._test_img_a),
                ops.convert_to_numpy(_test_img_b),
                data_range=1.0,
                channel_axis=2,
                window_size=3,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ) / 2
        p_ssim = self.ssim(self._test_img_a, _test_img_b)
        testing.assert_almost_equal(
            t_ssim, ops.convert_to_numpy(p_ssim), decimal=self.decimal
        )

    # SSIM same with small noise
    def test_ssim_noiser(self):
        _test_img_b = ops.clip(
            ops.add(
                self._test_img_a,
                ops.convert_to_tensor(
                    random.uniform(
                        (224, 224, 3),
                        minval=-0.3,
                        maxval=0.3,
                        dtype=K.floatx(),
                        seed=456,
                    )
                ),
            ),
            0.0,
            1.0,
        )
        t_ssim = (
            1
            - compare_ssim(
                ops.convert_to_numpy(self._test_img_a),
                ops.convert_to_numpy(_test_img_b),
                data_range=1.0,
                channel_axis=2,
                window_size=3,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ) / 2
        p_ssim = self.ssim(self._test_img_a, _test_img_b)
        testing.assert_almost_equal(
            t_ssim, ops.convert_to_numpy(p_ssim), decimal=self.decimal
        )

    # SSIM same with small noise
    def test_ssim_noisest(self):
        _test_img_b = ops.clip(
            ops.add(
                self._test_img_a,
                ops.convert_to_tensor(
                    random.uniform(
                        (224, 224, 3),
                        minval=-0.5,
                        maxval=0.5,
                        dtype=K.floatx(),
                        seed=456,
                    )
                ),
            ),
            0.0,
            1.0,
        )
        t_ssim = (
            1
            - compare_ssim(
                ops.convert_to_numpy(self._test_img_a),
                ops.convert_to_numpy(_test_img_b),
                data_range=1.0,
                channel_axis=2,
                window_size=3,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ) / 2
        p_ssim = self.ssim(self._test_img_a, _test_img_b)
        testing.assert_almost_equal(
            t_ssim, ops.convert_to_numpy(p_ssim), decimal=self.decimal
        )

    # SSIM different
    def test_ssim_different(self):
        _test_img_b = ops.convert_to_tensor(
            random.uniform(
                (224, 224, 3), minval=0, maxval=1.0, dtype=K.floatx(), seed=456
            )
        )
        t_ssim = (
            1
            - compare_ssim(
                ops.convert_to_numpy(self._test_img_a),
                ops.convert_to_numpy(_test_img_b),
                data_range=1.0,
                channel_axis=2,
                window_size=3,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ) / 2
        p_ssim = self.ssim(self._test_img_a, _test_img_b)
        testing.assert_almost_equal(
            t_ssim, ops.convert_to_numpy(p_ssim), decimal=self.decimal
        )


if __name__ == "__main__":
    unittest.main()
