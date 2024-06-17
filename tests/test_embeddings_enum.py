import unittest
from keras import utils, KerasTensor, ops
import numpy as np

from auramask.losses.embeddistance import cosine_distance
from auramask.models.face_embeddings import FaceEmbedEnum
from deepface.modules.representation import represent
from deepface.modules.preprocessing import load_image


class TestArcFaceEmbeddingEnum(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = FaceEmbedEnum.ARCFACE.get_model()
        self._embed_model_name = "ArcFace"
        self._embed_model_norm = "ArcFace"
        self._image_shape = (256, 256, 3)
        self.atol = 1e-2
        self.rtol = 1e-2
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        del self._embed_model_name
        del self._embed_model_norm
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/lfw.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)


class TestVGGFaceEmbeddingEnum(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = FaceEmbedEnum.VGGFACE.get_model()
        self._embed_model_name = "VGG-Face"
        self._embed_model_norm = "VGGFace"
        self._image_shape = (256, 256, 3)
        self.atol = 1e-2
        self.rtol = 1e-2
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        del self._embed_model_name
        del self._embed_model_norm
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/lfw.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)


class TestFaceNetEmbeddingEnum(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = FaceEmbedEnum.FACENET.get_model()
        self._embed_model_name = "Facenet"
        self._embed_model_norm = "Facenet"
        self._image_shape = (256, 256, 3)
        self.atol = 1e-2
        self.rtol = 1e-2
        return super().setUpClass()

    def tearDown(self) -> None:
        del self._image_shape
        del self._embed_model
        return super().tearDownClass()

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_lfw(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/lfw.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_same_embed_vggface(self):
        a: KerasTensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_near(dist, 0.0, rtol=self.rtol, atol=self.atol)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/fdf.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        np.testing.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a) / 255.0

        my_embed = self._embed_model(ops.expand_dims(a, axis=0))

        b, _ = load_image("./tests/tst_imgs/vggface2.png")

        df_embed = represent(
            b,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )
        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        self.assertGreater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
