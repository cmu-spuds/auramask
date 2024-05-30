from enum import Enum
from typing import Tuple
from datasets import load_dataset, Dataset
from auramask.utils import preprocessing
from os import cpu_count
import tensorflow as tf
from keras.preprocessing.image import img_to_array


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ("image", "image"))
    INSTAGRAM = ("logasja/lfw", "aug", ("orig", "aug"))
    FDF256 = ("logasja/FDF", "default", ("image", "image"))
    FDF = ("logasja/FDF", "fdf", ("image", "image"))
    VGGFACE2 = ("logasja/VGGFace2", "default", ("image", "image"))

    def fetch_dataset(
        self, t_split: str, v_split: str, ds_format: str = "tf"
    ) -> Tuple[Dataset, Dataset]:
        dataset, name, _ = self.value
        t_ds: Dataset = load_dataset(
            dataset, name, split=t_split, num_proc=cpu_count() if cpu_count() < 9 else 8
        )

        v_ds: Dataset = load_dataset(
            dataset, name, split=v_split, num_proc=cpu_count() if cpu_count() < 9 else 8
        )
        return t_ds, v_ds

    def get_data_loader(self, w: int, h: int, augment: bool = True):
        X, Y = self.value[2]
        loader = preprocessing.gen_image_loading_layers(w, h, crop=True)

        if augment:
            geom_aug = preprocessing.gen_geometric_aug_layers(
                augs_per_image=1, rate=0.5
            )
            augmenter = preprocessing.gen_non_geometric_aug_layers(
                augs_per_image=1, magnitude=0.2
            )

            def transforms(example):
                example[X] = tf.ragged.stack(
                    [tf.ragged.constant(img_to_array(image)) for image in example[X]]
                )
                if X == Y:
                    x = loader(example[X])
                    y = tf.identity(x)
                else:
                    x = loader(example[X])
                    example[Y] = tf.ragged.stack(
                        [
                            tf.ragged.constant(img_to_array(image))
                            for image in example[Y]
                        ]
                    )
                    y = loader(example[Y])
                data = geom_aug(
                    {"images": x, "segmentation_masks": y}
                )  # Geometric augmentations
                y = data["segmentation_masks"]  # Separate out target
                x = augmenter(data["images"])  # Pixel-level modifications
                return {"x": x, "y": y}
        else:

            def transforms(example):
                example[X] = tf.ragged.stack(
                    [tf.ragged.constant(img_to_array(image)) for image in example[X]]
                )
                if X == Y:
                    x = loader(example[X])
                    y = tf.identity(x)
                else:
                    x = loader(example[X])
                    example[Y] = tf.ragged.stack(
                        [
                            tf.ragged.constant(img_to_array(image))
                            for image in example[Y]
                        ]
                    )
                    y = loader(example[Y])
                return {"x": x, "y": y}

        return transforms
