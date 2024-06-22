# To test run docker `docker run -p 5005:5000 serengil/deepface`

import unittest
from keras import utils, ops, KerasTensor

import requests
import base64

import numpy as np

from auramask.losses.embeddistance import cosine_distance
from auramask.models.arcface import ArcFace
from auramask.models.deepid import DeepID

from auramask.models.facenet import FaceNet
from auramask.models.vggface import VggFace
from auramask.utils.preprocessing import rgb_to_bgr

FDF_IMG = "./tests/tst_imgs/fdf.jpg"
LFW_IMG = "./tests/tst_imgs/lfw.jpg"
VGG_IMG = "./tests/tst_imgs/vggface2.jpg"

PREPROC = False


def represent(
    img_path: str,
    model_name: str = "VGG-Face",
    align: bool = False,
    detector_backend: str = "mtcnn",
    enforce_detection: bool = True,
    normalization: str = "base",
):
    with open(img_path, "rb") as file:
        file_encode = base64.b64encode(file.read()).decode()
        dataurl = f"data:image/jpg;base64,{file_encode}"
        x = requests.post(
            "http://localhost:5005/represent",
            json={
                "img": dataurl,
                "model_name": model_name,
                "enforce_detection": enforce_detection,
                "detector_backend": detector_backend,
                "align": align,
            },
        )
    js = x.json()
    return js["results"]


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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)
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
                    FDF_IMG,
                    LFW_IMG,
                    VGG_IMG,
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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))

        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            LFW_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)


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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)
        a = ops.expand_dims(a, axis=0)
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a)

        np.testing.assert_equal(a[..., 0], a_transformed[..., 0] + 93.540)
        np.testing.assert_equal(a[..., 1], a_transformed[..., 1] + 104.7624)
        np.testing.assert_equal(a[..., 2], a_transformed[..., 2] + 129.1863)

    def test_preprocess_input_batch(self):
        a: KerasTensor = ops.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    FDF_IMG,
                    LFW_IMG,
                    VGG_IMG,
                ]
            ]
        )
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        np.testing.assert_equal(a[..., 0], a_transformed[..., 0] + 93.540)
        np.testing.assert_equal(a[..., 1], a_transformed[..., 1] + 104.7624)
        np.testing.assert_equal(a[..., 2], a_transformed[..., 2] + 129.1863)

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            LFW_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)


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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)
        a = ops.expand_dims(a, axis=0)
        from auramask.models.facenet import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        a_mean = ops.mean(a, axis=[-3, -2, -1])
        a_std = ops.std(a, axis=[-3, -2, -1])

        np.testing.assert_equal((ops.max(a) - a_mean) / a_std, ops.max(a_transformed))
        np.testing.assert_equal((ops.min(a) - a_mean) / a_std, ops.min(a_transformed))

        np.testing.assert_almost_equal(ops.mean(a_transformed, axis=[-3, -2, -1]), [0])
        np.testing.assert_almost_equal(ops.std(a_transformed, axis=[-3, -2, -1]), [1])
        self.assertEqual(ops.std(a, axis=[-3, -2, -1]).shape, (1,))

    def test_preprocess_input_batch(self):
        a: KerasTensor = ops.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    FDF_IMG,
                    LFW_IMG,
                    VGG_IMG,
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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            LFW_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)


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
        a: KerasTensor = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            LFW_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_almost_equal(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            FDF_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img(LFW_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img(FDF_IMG, target_size=(self._image_shape))
        a = rgb_to_bgr(utils.img_to_array(a) / 255.0)

        b = utils.load_img(VGG_IMG, target_size=(self._image_shape))
        b = rgb_to_bgr(utils.img_to_array(b) / 255.0)

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        df_embed = represent(
            VGG_IMG,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreaterEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
