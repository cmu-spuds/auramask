# ruff: noqa: E402

# To test run docker `docker run -p 5005:5000 serengil/deepface`
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from keras import utils, ops, KerasTensor, layers, config, backend
from PIL import Image

import requests
import cv2

import numpy as np

from auramask.losses.embeddistance import cosine_distance
from auramask.models.arcface import ArcFace
from auramask.models.deepid import DeepID

from auramask.models.facenet import FaceNet
from auramask.models.openface import OpenFace
from auramask.models.vggface import VggFace


def rgb_to_bgr(x):
    if backend.image_data_format() == "channels_first":
        # 'RGB'->'BGR'
        if backend.ndim(x) == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    return x


config.disable_traceback_filtering()
backend.set_image_data_format("channels_last")

FDF_IMG = "https://github.com/cmu-spuds/auramask/blob/0c9e679bfeb532b8998955da8064c51064768483/tests/tst_imgs/fdf.jpg?raw=True"
LFW_IMG = "https://github.com/cmu-spuds/auramask/blob/0c9e679bfeb532b8998955da8064c51064768483/tests/tst_imgs/lfw.jpg?raw=True"
VGG_IMG = "https://github.com/cmu-spuds/auramask/blob/0c9e679bfeb532b8998955da8064c51064768483/tests/tst_imgs/vggface2.jpg?raw=True"

PREPROC = False


def represent(
    img_path: str,
    model_name: str = "VGG-Face",
    align: bool = False,
    detector_backend: str = "mtcnn",
    enforce_detection: bool = True,
    normalization: str = "base",
):
    x = requests.post(
        "http://localhost:5005/represent",
        json={
            "img": img_path,
            "model_name": model_name,
            "enforce_detection": enforce_detection,
            "detector_backend": detector_backend,
            "align": align,
        },
    )
    js = x.json()
    return js["results"]


def test_same_embed_batch(obj: unittest.TestCase, img_paths: list):
    rs = layers.Rescaling(scale=1.0 / 255)
    rsiz = layers.Resizing(obj._image_shape[0], obj._image_shape[1])

    a = []
    for img_path in img_paths:
        file = utils.get_file(origin=img_path, cache_subdir="tst_imgs")
        img = utils.load_img(file)
        a.append(rsiz(utils.img_to_array(img)))

    a = ops.stack(a)

    a = rs(rsiz(a))

    my_embed = obj._embed_model(a, training=False)

    df_embed = []
    for img_path in img_paths:
        resp = represent(
            img_path,
            model_name=obj._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=obj._embed_model_norm,
        )

        df_embed.append(resp[0]["embedding"])

    dist = cosine_distance(my_embed, df_embed, axis=-1)
    dist = dist.detach().cpu()
    np.testing.assert_allclose(dist, np.zeros_like(dist), atol=obj.atol, rtol=obj.rtol)
    # obj.assertAlmostEqual(float(dist), 0.0, places=4)


def test_same_embed(obj: unittest.TestCase, img_path: str):
    rs = layers.Rescaling(scale=1.0 / 255)
    rsiz = layers.Resizing(obj._image_shape[0], obj._image_shape[1])

    file = utils.get_file(origin=img_path, cache_subdir="tst_imgs")
    a = utils.load_img(file)
    a = utils.img_to_array(a)
    a = rs(rsiz(a))
    # utils.array_to_img(a).save("Altered.png")

    my_embed = obj._embed_model(ops.expand_dims(a, axis=0), training=False)

    df_embed = represent(
        img_path,
        model_name=obj._embed_model_name,
        enforce_detection=False,
        align=False,
        detector_backend="skip",
        normalization=obj._embed_model_norm,
    )

    df_embed = df_embed[0]["embedding"]

    dist = cosine_distance(my_embed[0], df_embed, axis=-1)
    # np.testing.assert_allclose(my_embed[0], df_embed, atol=obj.atol, rtol=obj.rtol)
    obj.assertAlmostEqual(float(dist), 0.0, places=4)


def test_diff_embed(obj: unittest.TestCase, img_path_a, img_path_b):
    rs = layers.Rescaling(scale=1.0 / 255)
    rsiz = layers.Resizing(obj._image_shape[0], obj._image_shape[1])

    print(obj._image_shape)

    print(rs)
    print(rsiz)

    file_a = utils.get_file(origin=img_path_a, cache_subdir="tst_imgs")
    a = rs(rsiz(cv2.imread(file_a)))

    my_embed = obj._embed_model(ops.expand_dims(a, axis=0), training=False)

    df_embed = represent(
        img_path_b,
        model_name=obj._embed_model_name,
        enforce_detection=False,
        align=False,
        detector_backend="skip",
        normalization=obj._embed_model_norm,
    )

    df_embed = df_embed[0]["embedding"]

    dist = cosine_distance(my_embed, df_embed, axis=-1)
    obj.assertGreaterEqual(dist, 0.0)


class TestArcFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = ArcFace(preprocess=PREPROC)
        self._embed_model_name = "ArcFace"
        self._embed_model_norm = "ArcFace"
        self._image_shape = (112, 112, 3)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        del self._embed_model_name
        del self._embed_model_norm
        return super().tearDownClass()

    def test_preprocess_input(self):
        file = utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs")
        a: Image = utils.load_img(file, target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0
        a = ops.expand_dims(a, axis=0)
        from auramask.models.arcface import preprocess_input

        a_transformed = preprocess_input(a)

        np.testing.assert_equal((ops.max(a) - 127.5) / 128.0, ops.max(a_transformed))
        np.testing.assert_equal((ops.min(a) - 127.5) / 128.0, ops.min(a_transformed))

    def test_preprocess_input_batch(self):
        a: KerasTensor = ops.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=LFW_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=VGG_IMG, cache_subdir="tst_imgs"),
                ]
            ]
        )
        from auramask.models.arcface import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        for i in range(0, 3):
            np.testing.assert_equal(
                (ops.max(a[i]) - 127.5) / 128.0, ops.max(a_transformed[i])
            )
            np.testing.assert_equal(
                (ops.min(a[i]) - 127.5) / 128.0, ops.min(a_transformed[i])
            )

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        test_same_embed(self, FDF_IMG)

    def test_same_embed_lfw(self):
        test_same_embed(self, LFW_IMG)

    def test_same_embed_vggface(self):
        test_same_embed(self, VGG_IMG)

    def test_same_embed_batch(self):
        test_same_embed_batch(self, [FDF_IMG, LFW_IMG, VGG_IMG])

    def test_diff_embed_lfw_fdf(self):
        test_diff_embed(self, LFW_IMG, FDF_IMG)

    def test_diff_embed_lfw_vgg(self):
        test_diff_embed(self, LFW_IMG, VGG_IMG)

    def test_diff_embed_fdf_vgg(self):
        test_diff_embed(self, FDF_IMG, VGG_IMG)


class TestVGGFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = VggFace(preprocess=PREPROC, include_top=False)
        self._embed_model_name = "VGG-Face"
        self._embed_model_norm = "VGGFace"
        self._image_shape = (224, 224, 3)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        del self._embed_model_name
        del self._embed_model_norm
        return super().tearDownClass()

    def test_preprocess_input(self):
        file = utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs")
        a: Image = utils.load_img(file, target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0
        a = ops.expand_dims(a, axis=0)
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a)

        np.testing.assert_allclose(
            a[..., 0], a_transformed[..., 0] + 93.540, atol=self.atol, rtol=self.rtol
        )
        np.testing.assert_allclose(
            a[..., 1], a_transformed[..., 1] + 104.7624, atol=self.atol, rtol=self.rtol
        )
        np.testing.assert_allclose(
            a[..., 2], a_transformed[..., 2] + 129.1863, atol=self.atol, rtol=self.rtol
        )

    def test_preprocess_input_batch(self):
        a: KerasTensor = ops.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=LFW_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=VGG_IMG, cache_subdir="tst_imgs"),
                ]
            ]
        )
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        np.testing.assert_allclose(
            a[..., 0], a_transformed[..., 0] + 93.540, atol=self.atol, rtol=self.rtol
        )
        np.testing.assert_allclose(
            a[..., 1], a_transformed[..., 1] + 104.7624, atol=self.atol, rtol=self.rtol
        )
        np.testing.assert_allclose(
            a[..., 2], a_transformed[..., 2] + 129.1863, atol=self.atol, rtol=self.rtol
        )

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        test_same_embed(self, FDF_IMG)

    def test_same_embed_lfw(self):
        test_same_embed(self, LFW_IMG)

    def test_same_embed_vggface(self):
        test_same_embed(self, VGG_IMG)

    def test_same_embed_batch(self):
        test_same_embed_batch(self, [FDF_IMG, LFW_IMG, VGG_IMG])

    def test_diff_embed_lfw_fdf(self):
        test_diff_embed(self, LFW_IMG, FDF_IMG)

    def test_diff_embed_lfw_vgg(self):
        test_diff_embed(self, LFW_IMG, VGG_IMG)

    def test_diff_embed_fdf_vgg(self):
        test_diff_embed(self, FDF_IMG, VGG_IMG)


class TestFaceNetEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = FaceNet(preprocess=PREPROC)
        self._embed_model_name = "Facenet"
        self._embed_model_norm = "Facenet"
        self._image_shape = (160, 160, 3)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        return super().tearDownClass()

    def test_preprocess_input(self):
        file = utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs")
        a: Image = utils.load_img(file, target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0
        a = ops.expand_dims(a, axis=0)
        from auramask.models.facenet import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        a_mean = ops.mean(a, axis=[-3, -2, -1])
        a_std = ops.std(a, axis=[-3, -2, -1])

        np.testing.assert_equal((ops.max(a) - a_mean) / a_std, ops.max(a_transformed))
        np.testing.assert_equal((ops.min(a) - a_mean) / a_std, ops.min(a_transformed))

        np.testing.assert_almost_equal(ops.mean(a_transformed, axis=[-3, -2, -1]), [0])
        np.testing.assert_almost_equal(
            ops.std(a_transformed, axis=[-3, -2, -1]), [1], decimal=5
        )
        self.assertEqual(ops.std(a, axis=[-3, -2, -1]).shape, (1,))

    def test_preprocess_input_batch(self):
        a: KerasTensor = ops.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    utils.get_file(origin=FDF_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=LFW_IMG, cache_subdir="tst_imgs"),
                    utils.get_file(origin=VGG_IMG, cache_subdir="tst_imgs"),
                ]
            ]
        )

        from auramask.models.facenet import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        a_mean = ops.mean(a, axis=[-3, -2, -1])
        a_std = ops.std(a, axis=[-3, -2, -1])

        for i in range(0, 3):
            np.testing.assert_equal(
                (ops.max(a[i]) - a_mean[i]) / a_std[i],
                ops.max(a_transformed[i]),
            )
            np.testing.assert_equal(
                (ops.min(a[i]) - a_mean[i]) / a_std[i],
                ops.min(a_transformed[i]),
            )

        np.testing.assert_almost_equal(
            ops.mean(a_transformed, axis=[-3, -2, -1]), [0.0, 0.0, 0.0]
        )
        np.testing.assert_almost_equal(
            ops.std(a_transformed, axis=[-3, -2, -1]), [1.0, 1.0, 1.0]
        )

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        test_same_embed(self, FDF_IMG)

    def test_same_embed_lfw(self):
        test_same_embed(self, LFW_IMG)

    def test_same_embed_vggface(self):
        test_same_embed(self, VGG_IMG)

    def test_diff_embed_lfw_fdf(self):
        test_diff_embed(self, LFW_IMG, FDF_IMG)

    def test_same_embed_batch(self):
        test_same_embed_batch(self, [FDF_IMG, LFW_IMG, VGG_IMG])

    def test_diff_embed_lfw_vgg(self):
        test_diff_embed(self, LFW_IMG, VGG_IMG)

    def test_diff_embed_fdf_vgg(self):
        test_diff_embed(self, FDF_IMG, VGG_IMG)


class TestDeepIDEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = DeepID(preprocess=PREPROC)
        self._embed_model_name = "DeepID"
        self._embed_model_norm = "base"
        self._image_shape = (55, 47, 3)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        test_same_embed(self, FDF_IMG)

    def test_same_embed_lfw(self):
        test_same_embed(self, LFW_IMG)

    def test_same_embed_vggface(self):
        test_same_embed(self, VGG_IMG)

    def test_same_embed_batch(self):
        test_same_embed_batch(self, [FDF_IMG, LFW_IMG, VGG_IMG])

    def test_diff_embed_lfw_fdf(self):
        test_diff_embed(self, LFW_IMG, FDF_IMG)

    def test_diff_embed_lfw_vgg(self):
        test_diff_embed(self, LFW_IMG, VGG_IMG)

    def test_diff_embed_fdf_vgg(self):
        test_diff_embed(self, FDF_IMG, VGG_IMG)


class TestOpenFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = OpenFace(preprocess=PREPROC)
        self._embed_model_name = "OpenFace"
        self._embed_model_norm = "base"
        self._image_shape = (96, 96, 3)
        self.atol = 1.2e-4
        self.rtol = 1.2e-4
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        test_same_embed(self, FDF_IMG)

    def test_same_embed_lfw(self):
        test_same_embed(self, LFW_IMG)

    def test_same_embed_vggface(self):
        test_same_embed(self, VGG_IMG)

    def test_same_embed_batch(self):
        test_same_embed_batch(self, [FDF_IMG, LFW_IMG, VGG_IMG])

    def test_diff_embed_lfw_fdf(self):
        test_diff_embed(self, LFW_IMG, FDF_IMG)

    def test_diff_embed_lfw_vgg(self):
        test_diff_embed(self, LFW_IMG, VGG_IMG)

    def test_diff_embed_fdf_vgg(self):
        test_diff_embed(self, FDF_IMG, VGG_IMG)


if __name__ == "__main__":
    unittest.main()
