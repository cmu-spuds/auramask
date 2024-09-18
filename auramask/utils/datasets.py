from enum import Enum
from typing import TypedDict
from datasets import load_dataset
from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.utils import preprocessing
from os import cpu_count
from keras import ops, utils
import numpy as np
import PIL


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ["image"])
    INSTAGRAM = ("logasja/lfw", "aug", ["orig", "aug"])
    FDF256 = ("logasja/FDF", "default", ["image"])
    FDF = ("logasja/FDF", "fdf", ["image"])
    VGGFACE2 = ("logasja/VGGFace2", "256", ["image"])

    def fetch_dataset(self, t_split: str, v_split: str):
        dataset, name, _ = self.value
        t_ds = load_dataset(
            dataset,
            name,
            split=t_split,
            num_proc=cpu_count() if cpu_count() < 9 else 8,
        )

        v_ds = load_dataset(
            dataset,
            name,
            split=v_split,
            num_proc=cpu_count() if cpu_count() < 9 else 8,
        )

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
        loader = preprocessing.gen_image_loading_layers(**args)

        if isinstance(features, dict):  # case batch_size=None: nothing to collate
            batch = {}
            for k, values in features.items():
                first = values[0]
                if isinstance(first, np.ndarray):
                    batch[k] = values
                elif ops.is_tensor(first):
                    batch[k] = np.stack(ops.convert_to_numpy(values))
                elif PIL.Image.isImageType(first):
                    batch[k] = np.stack([utils.img_to_array(v) for v in values])
                else:
                    batch[k] = np.array([v for v in values])
        elif ops.is_tensor(features):
            batch = {"image": ops.convert_to_numpy(features)}
        elif PIL.Image.isImageType(features):
            batch = {"image": utils.img_to_array(features)}
        else:
            first = features[0]
            batch = {}
            for k, v in first.items():
                if isinstance(v, np.ndarray):
                    batch[k] = np.stack([loader(image=f[k])["image"] for f in features])
                elif ops.is_tensor(v):
                    batch[k] = np.stack(
                        [
                            loader(image=ops.convert_to_numpy(f[k]))["image"]
                            for f in features
                        ]
                    )
                elif PIL.Image.isImageType(v):
                    batch[k] = np.stack(
                        [
                            loader(image=utils.img_to_array(f[k]))["image"]
                            for f in features
                        ]
                    )
                else:
                    batch[k] = np.array([f[k] for f in features])
        del loader
        return batch

    @staticmethod
    def compute_embeddings(img_batch, embedding_models: list[FaceEmbedEnum]) -> dict:
        features = []
        if embedding_models:
            for model in embedding_models:
                embed = model.get_model()(img_batch)
                features.append(embed)

        return tuple(features)

    def data_augmenter(
        self, examples, geom, aug, embedding_models: list[FaceEmbedEnum]
    ):
        cols = self.value[-1]
        x = examples[cols[0]]

        # Determine if desired output is a referenced output or the original image
        if len(cols) > 1:
            y = examples[cols[1]]
        elif "target" in examples.keys():
            y = examples["target"]
        else:
            y = ops.copy(x)  # Separate out target

        # TODO: Make sure when geometric is applied it is applied the same to x and y
        a = [
            geom(image=i, mask=j) for i, j in zip(x, y)
        ]  # Apply geometric modifications
        x, y = np.stack([i["image"] for i in a]), ops.stack([j["mask"] for j in a])

        emb = self.compute_embeddings(x, embedding_models)
        x = ops.stack([aug(image=i)["image"] for i in x])  # Pixel-level modifications
        return (x, (y, emb))
