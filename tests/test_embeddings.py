import unittest
import tensorflow as tf
from keras import utils

from auramask.losses.embeddistance import cosine_distance
from auramask.models.arcface import ArcFace
from deepface.modules.representation import represent

from auramask.models.facenet import FaceNet
from auramask.models.vggface import VggFace

tf.config.run_functions_eagerly(True)


class TestArcFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = ArcFace(preprocess=True)
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
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)
        a = tf.expand_dims(a, axis=0)
        from auramask.models.arcface import preprocess_input

        a_transformed = preprocess_input(a)

        tf.debugging.assert_equal(
            (tf.reduce_max(a) - 127.5) / 128.0, tf.reduce_max(a_transformed)
        )
        tf.debugging.assert_equal(
            (tf.reduce_min(a) - 127.5) / 128.0, tf.reduce_min(a_transformed)
        )

    def test_preprocess_input_batch(self):
        a: tf.Tensor = tf.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    "./tests/tst_imgs/fdf.png",
                    "./tests/tst_imgs/lfw.png",
                    "./tests/tst_imgs/vggface2.png",
                ]
            ]
        )
        from auramask.models.arcface import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        for i in range(0, 3):
            tf.debugging.assert_equal(
                (tf.reduce_max(a[i]) - 127.5) / 128.0, tf.reduce_max(a_transformed[i])
            )
            tf.debugging.assert_equal(
                (tf.reduce_min(a[i]) - 127.5) / 128.0, tf.reduce_min(a_transformed[i])
            )

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_lfw(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_vggface(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)


class TestVGGFaceEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = VggFace(preprocess=True, include_top=False)
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
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)
        a = tf.expand_dims(a, axis=0)
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a)

        tf.debugging.assert_equal(a[..., 0], a_transformed[..., 0] + 93.540)
        tf.debugging.assert_equal(a[..., 1], a_transformed[..., 1] + 104.7624)
        tf.debugging.assert_equal(a[..., 2], a_transformed[..., 2] + 129.1863)

    def test_preprocess_input_batch(self):
        a: tf.Tensor = tf.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    "./tests/tst_imgs/fdf.png",
                    "./tests/tst_imgs/lfw.png",
                    "./tests/tst_imgs/vggface2.png",
                ]
            ]
        )
        from auramask.models.vggface import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        tf.debugging.assert_equal(a[..., 0], a_transformed[..., 0] + 93.540)
        tf.debugging.assert_equal(a[..., 1], a_transformed[..., 1] + 104.7624)
        tf.debugging.assert_equal(a[..., 2], a_transformed[..., 2] + 129.1863)

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_lfw(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_vggface(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)


class TestFaceNetEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self._embed_model = FaceNet(preprocess=True)
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
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)
        a = tf.expand_dims(a, axis=0)
        from auramask.models.facenet import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        a_mean = tf.reduce_mean(a, axis=[-3, -2, -1])
        a_std = tf.math.reduce_std(a, axis=[-3, -2, -1])

        tf.debugging.assert_equal(
            (tf.reduce_max(a) - a_mean) / a_std, tf.reduce_max(a_transformed)
        )
        tf.debugging.assert_equal(
            (tf.reduce_min(a) - a_mean) / a_std, tf.reduce_min(a_transformed)
        )

        tf.debugging.assert_near(tf.reduce_mean(a_transformed, axis=[-3, -2, -1]), [0])
        tf.debugging.assert_near(
            tf.math.reduce_std(a_transformed, axis=[-3, -2, -1]), [1]
        )
        self.assertEqual(tf.math.reduce_std(a, axis=[-3, -2, -1]).shape, (1,))

    def test_preprocess_input_batch(self):
        a: tf.Tensor = tf.stack(
            [
                utils.img_to_array(utils.load_img(pth, target_size=(self._image_shape)))
                for pth in [
                    "./tests/tst_imgs/fdf.png",
                    "./tests/tst_imgs/lfw.png",
                    "./tests/tst_imgs/vggface2.png",
                ]
            ]
        )
        from auramask.models.facenet import preprocess_input

        a_transformed = preprocess_input(a).numpy()

        a_mean = tf.reduce_mean(a, axis=[-3, -2, -1])
        a_std = tf.math.reduce_std(a, axis=[-3, -2, -1])

        for i in range(0, 3):
            tf.debugging.assert_equal(
                (tf.reduce_max(a[i]) - a_mean[i]) / a_std[i],
                tf.reduce_max(a_transformed[i]),
            )
            tf.debugging.assert_equal(
                (tf.reduce_min(a[i]) - a_mean[i]) / a_std[i],
                tf.reduce_min(a_transformed[i]),
            )

        tf.debugging.assert_near(
            tf.reduce_mean(a_transformed, axis=[-3, -2, -1]), [0.0, 0.0, 0.0]
        )
        tf.debugging.assert_near(
            tf.math.reduce_std(a_transformed, axis=[-3, -2, -1]), [1.0, 1.0, 1.0]
        )

    # Test Same CD: 0
    def test_same_embed_fdf(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/fdf.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_lfw(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/lfw.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_same_embed_vggface(self):
        a: tf.Tensor = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        a = utils.img_to_array(a)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

        df_embed = represent(
            a,
            model_name=self._embed_model_name,
            enforce_detection=False,
            align=False,
            detector_backend="skip",
            normalization=self._embed_model_norm,
        )

        df_embed = df_embed[0]["embedding"]

        dist = cosine_distance(my_embed, df_embed, axis=-1)
        tf.debugging.assert_near(dist, 0.0)

    def test_diff_embed_lfw_fdf(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_lfw_vgg(self):
        a = utils.load_img("./tests/tst_imgs/lfw.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)

    def test_diff_embed_fdf_vgg(self):
        a = utils.load_img("./tests/tst_imgs/fdf.png", target_size=(self._image_shape))
        a = utils.img_to_array(a)

        b = utils.load_img(
            "./tests/tst_imgs/vggface2.png", target_size=(self._image_shape)
        )
        b = utils.img_to_array(b)

        my_embed = self._embed_model(tf.expand_dims(a, axis=0))

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
        tf.debugging.assert_greater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
