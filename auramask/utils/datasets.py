from enum import Enum
from typing import TypedDict
from datasets import load_dataset
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils import preprocessing
from os import cpu_count
from keras import ops


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ["image"])
    INSTAGRAM = ("logasja/lfw", "aug", ["orig", "aug"])
    FDF256 = ("logasja/FDF", "default", ["image"])
    FDF = ("logasja/FDF", "fdf", ["image"])
    VGGFACE2 = ("logasja/VGGFace2", "256", ["image"])

    def fetch_dataset(self, t_split: str, v_split: str, ds_format="tensorflow"):
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
    def get_augmenters(geom_config: GeomConfig, aug_config: AugConfig):
        return {
            "geom": preprocessing.gen_geometric_aug_layers(**geom_config),
            "aug": preprocessing.gen_non_geometric_aug_layers(**aug_config),
        }

    @staticmethod
    def data_collater(features, args: LoaderConfig):
        import numpy as np

        loader = preprocessing.gen_image_loading_layers(**args)

        if isinstance(features, dict):  # case batch_size=None: nothing to collate
            batch = features
        elif ops.is_tensor(features):
            batch = {"image": loader(features)}
        else:
            first = features[0]
            batch = {}
            for k, v in first.items():
                if isinstance(v, np.ndarray):
                    batch[k] = loader(
                        ops.stack([ops.convert_to_tensor(f[k]) for f in features])
                    )
                elif ops.is_tensor(v):
                    batch[k] = loader(ops.stack([f[k] for f in features]))
                else:
                    batch[k] = np.array([f[k] for f in features])
        del loader
        return batch

    @staticmethod
    def compute_embeddings(img_batch, embedding_models: list[FaceEmbedEnum]) -> dict:
        features = []
        names = []
        if embedding_models:
            for model in embedding_models:
                embed = model.get_model()(img_batch)
                features.append(embed)
                names.append(model.name)

        return tuple(features)

    @staticmethod
    def data_augmenter(examples, geom, aug):
        data = geom(examples)  # Geometric augmentations
        y = data  # Separate out target
        x = aug(data)  # Pixel-level modifications
        return (x, y)
