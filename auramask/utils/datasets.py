from enum import Enum
from typing import Tuple
from datasets import load_dataset, Dataset
from auramask.utils import preprocessing
from os import cpu_count
from typing import TypedDict


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ["image"])
    INSTAGRAM = ("logasja/lfw", "aug", ["orig", "aug"])
    FDF256 = ("logasja/FDF", "default", ["image"])
    FDF = ("logasja/FDF", "fdf", ["image"])
    VGGFACE2 = ("logasja/VGGFace2", "256", ["image"])

    def fetch_dataset(
        self, t_split: str, v_split: str, ds_format="tf"
    ) -> Tuple[Dataset, Dataset]:
        dataset, name, _ = self.value
        t_ds = load_dataset(
            dataset,
            name,
            split=t_split,
            num_proc=cpu_count() if cpu_count() < 9 else 8,
        )

        t_ds.set_format(ds_format, columns=self.value[2])

        v_ds = load_dataset(
            dataset,
            name,
            split=v_split,
            num_proc=cpu_count() if cpu_count() < 9 else 8,
        )

        v_ds.set_format(ds_format, columns=self.value[2])

        return t_ds, v_ds

    class LoaderConfig(TypedDict):
        w: int
        h: int
        crop: bool

    class GeomConfig(TypedDict):
        augs_per_image: int
        rate: float

    class AugConfig(TypedDict):
        augs_per_image: int
        rate: float
        magnitude: float

    @staticmethod
    def get_augmenters(
        loader_config: LoaderConfig, geom_config: GeomConfig, aug_config: AugConfig
    ):
        return {
            "loader": preprocessing.gen_image_loading_layers(**loader_config),
            "geom": preprocessing.gen_geometric_aug_layers(**geom_config),
            "aug": preprocessing.gen_non_geometric_aug_layers(**aug_config),
        }

    @staticmethod
    def data_collater(features, loader):
        if isinstance(features, dict):  # case batch_size=None: nothing to collate
            return features
        # elif config.TF_AVAILABLE:
        #     import tensorflow as tf
        # else:
        #     raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        import tensorflow as tf
        import numpy as np

        first = features[0]
        batch = {}
        for k, v in first.items():
            if isinstance(v, np.ndarray):
                batch[k] = loader(np.stack([f[k] for f in features]))
            elif isinstance(v, tf.Tensor):
                batch[k] = loader(tf.stack([f[k] for f in features]))
            else:
                batch[k] = np.array([f[k] for f in features])
        return batch

    @staticmethod
    def data_augmenter(examples, geom, aug):
        data = geom(examples)  # Geometric augmentations
        y = data  # Separate out target
        x = aug(data)  # Pixel-level modifications
        return (x, y)
