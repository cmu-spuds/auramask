import unittest
import tensorflow as tf
from keras.layers import CenterCrop
from unittest.mock import patch, MagicMock
from datasets import Dataset
from auramask.utils.datasets import DatasetEnum
from numpy.random import rand
from PIL import Image


class testCenterCrop(CenterCrop):
    def call(self, inputs):
        return inputs


def create_PIL_Image(dims: tuple) -> Image:
    imarray = rand(dims[0], dims[1], dims[2]) * 255
    im = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    return im


class TestDatasetEnum(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUpClass()

    def tearDown(self) -> None:
        return super().tearDownClass()

    def test_enum_values(self):
        self.assertEqual(
            DatasetEnum.LFW.value, ("logasja/lfw", "default", ("image", "image"))
        )
        self.assertEqual(
            DatasetEnum.INSTAGRAM.value, ("logasja/lfw", "aug", ("orig", "aug"))
        )
        self.assertEqual(
            DatasetEnum.FDF256.value, ("logasja/FDF", "default", ("image", "image"))
        )
        self.assertEqual(
            DatasetEnum.FDF.value, ("logasja/FDF", "fdf", ("image", "image"))
        )
        self.assertEqual(
            DatasetEnum.VGGFACE2.value,
            ("logasja/VGGFace2", "default", ("image", "image")),
        )

    @patch("auramask.utils.datasets.load_dataset")
    @patch("auramask.utils.datasets.cpu_count")
    def test_fetch_lfw_dataset(self, mock_cpu_count, mock_load_dataset):
        mock_cpu_count.return_value = 4  # Assuming a system with 4 CPUs for this test
        mock_train_ds = MagicMock(spec=Dataset)
        mock_val_ds = MagicMock(spec=Dataset)
        mock_load_dataset.side_effect = [mock_train_ds, mock_val_ds]

        t_split = "train[:95%]"
        v_split = "train[96%:]"

        train_ds, val_ds = DatasetEnum.LFW.fetch_dataset(t_split, v_split)

        mock_load_dataset.assert_any_call(
            "logasja/lfw", "default", split=t_split, num_proc=4
        )
        mock_load_dataset.assert_any_call(
            "logasja/lfw", "default", split=v_split, num_proc=4
        )
        self.assertEqual(train_ds, mock_train_ds)
        self.assertEqual(val_ds, mock_val_ds)

    @patch("auramask.utils.datasets.load_dataset")
    @patch("auramask.utils.datasets.cpu_count")
    def test_fetch_vgg2_dataset(self, mock_cpu_count, mock_load_dataset):
        mock_cpu_count.return_value = 4  # Assuming a system with 4 CPUs for this test
        mock_train_ds = MagicMock(spec=Dataset)
        mock_val_ds = MagicMock(spec=Dataset)
        mock_load_dataset.side_effect = [mock_train_ds, mock_val_ds]

        t_split = "train"
        v_split = "test"

        train_ds, val_ds = DatasetEnum.VGGFACE2.fetch_dataset(t_split, v_split)

        mock_load_dataset.assert_any_call(
            "logasja/VGGFace2", "default", split=t_split, num_proc=4
        )
        mock_load_dataset.assert_any_call(
            "logasja/VGGFace2", "default", split=v_split, num_proc=4
        )
        self.assertEqual(train_ds, mock_train_ds)
        self.assertEqual(val_ds, mock_val_ds)

    @patch("auramask.utils.datasets.preprocessing")
    @patch("auramask.utils.datasets.img_to_array")
    def test_get_data_loader(self, mock_img_to_array, mock_preprocessing):
        w, h = 256, 256
        example_data = {"image": ["fake_image1", "fake_image2"]}

        # Mock the preprocessing functions
        mock_loader = MagicMock()
        mock_geom_aug = MagicMock()
        mock_augmenter = MagicMock()
        mock_preprocessing.gen_image_loading_layers.return_value = mock_loader
        mock_preprocessing.gen_geometric_aug_layers.return_value = mock_geom_aug
        mock_preprocessing.gen_non_geometric_aug_layers.return_value = mock_augmenter

        # Without augmentation
        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=False)
        transformed_example = data_loader(example_data)
        self.assertIn("x", transformed_example)
        self.assertIn("y", transformed_example)

        # With augmentation
        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=True)
        transformed_example = data_loader(example_data)
        self.assertIn("x", transformed_example)
        self.assertIn("y", transformed_example)

    @patch("auramask.utils.preprocessing.CenterCrop", testCenterCrop)
    def test_data_resizing_smaller(self):
        w, h = 256, 256
        example_data = {
            "image": [create_PIL_Image((64, 64, 3)), create_PIL_Image((64, 64, 3))]
        }

        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=False)
        transformed_example = data_loader(example_data)
        self.assertIn("x", transformed_example)
        tf.debugging.assert_rank(transformed_example["x"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["x"].shape)
        self.assertIn("y", transformed_example)
        tf.debugging.assert_rank(transformed_example["y"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["y"].shape)

    @patch("auramask.utils.preprocessing.CenterCrop", testCenterCrop)
    def test_data_resizing_larger(self):
        w, h = 256, 256
        example_data = {
            "image": [create_PIL_Image((512, 512, 3)), create_PIL_Image((512, 512, 3))]
        }

        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=False)
        transformed_example = data_loader(example_data)
        self.assertIn("x", transformed_example)
        tf.debugging.assert_rank(transformed_example["x"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["x"].shape)
        self.assertIn("y", transformed_example)
        tf.debugging.assert_rank(transformed_example["y"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["y"].shape)

    @patch("auramask.utils.preprocessing.CenterCrop", testCenterCrop)
    def test_data_resizing_same(self):
        w, h = 256, 256
        example_data = {
            "image": [create_PIL_Image((256, 256, 3)), create_PIL_Image((256, 256, 3))]
        }

        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=False)
        transformed_example = data_loader(example_data)
        self.assertIn("x", transformed_example)
        tf.debugging.assert_rank(transformed_example["x"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["x"].shape)
        self.assertIn("y", transformed_example)
        tf.debugging.assert_rank(transformed_example["y"], 4)
        tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["y"].shape)

    @patch("auramask.utils.preprocessing.CenterCrop", testCenterCrop)
    def test_data_resizing_mixed_smaller(self):
        w, h = 256, 256
        examples = [
            {"image": [create_PIL_Image((256, 256, 3)), create_PIL_Image((64, 64, 3))]},
            {"image": [create_PIL_Image((64, 64, 3)), create_PIL_Image((256, 256, 3))]},
        ]

        data_loader = DatasetEnum.LFW.get_data_loader(w, h, augment=False)
        for example_data in examples:
            transformed_example = data_loader(example_data)
            self.assertIn("x", transformed_example)
            tf.debugging.assert_rank(transformed_example["x"], 4)
            tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["x"].shape)
            self.assertIn("y", transformed_example)
            tf.debugging.assert_rank(transformed_example["y"], 4)
            tf.debugging.assert_equal((2, 256, 256, 3), transformed_example["y"].shape)


if __name__ == "__main__":
    unittest.main()
