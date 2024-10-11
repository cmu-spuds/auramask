from enum import Enum
import os
from typing import Callable, TypedDict
from datasets import load_dataset, Dataset, load_from_disk, features
from torch import NoneType
from auramask.utils import preprocessing
from os import cpu_count
from keras import utils, backend
import numpy as np
from albumentations import clahe
import PIL


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ["image"])
    INSTAGRAM = ("logasja/lfw", "aug", ["orig", "aug"])
    FDF256 = ("logasja/FDF", "default", ["image"])
    FDF = ("logasja/FDF", "fdf", ["image"])
    VGGFACE2 = ("logasja/VGGFace2", "256", ["image"])

    class LoaderConfig(TypedDict):
        w: int
        h: int
        crop: bool

    def fetch_dataset(self):
        dataset, name, _ = self.value
        ds = load_dataset(
            dataset,
            name,
            split="train",
            num_proc=cpu_count(),
        )

        return ds

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

        batch = {}

        if isinstance(features, list):  # Just one column (assume it is an image column)
            first = features[0]
            tmp_features = {}
            for k, values in first.items():
                tmp_features[k] = [i[k] for i in features]
            del features
            features = tmp_features

        for k, values in features.items():
            first = values[0]
            if isinstance(first, np.ndarray) and np.ndim(first) == 3:
                batch[k] = np.array([loader(image=i)["image"] for i in values])
            elif PIL.Image.isImageType(first):
                batch[k] = np.stack(
                    [
                        loader(image=utils.img_to_array(v, dtype="uint8"))["image"]
                        for v in values
                    ]
                )
            else:
                batch[k] = np.array(values)
        del loader
        return batch

    def data_augmenter(self, examples: list[dict] | dict, geom, aug):
        cols = self.value[-1]
        if isinstance(examples, list):
            x = np.stack([ex[cols[0]] for ex in examples], dtype="float32")
            # Determine if desired output is a referenced output or the original image
            if len(cols) > 1:
                y = np.stack([ex[cols[1]] for ex in examples], dtype="float32")
            elif "target" in examples[0].keys():
                y = np.stack([ex["target"] for ex in examples], dtype="float32")
            else:
                y = np.copy(x)  # Separate out target
        else:
            x = np.stack(examples[cols[0]], dtype="float32")
            # Determine if desired output is a referenced output or the original image
            if len(cols) > 1:
                y = np.stack(examples[cols[1]], dtype="float32")
            elif "target" in examples.keys():
                y = np.stack(examples["target"], dtype="float32")
            else:
                y = np.copy(x)  # Separate out target

        a = [
            geom(image=i, mask=j) for i, j in zip(x, y)
        ]  # Apply geometric modifications
        x, y = np.stack([i["image"] for i in a]), np.stack([j["mask"] for j in a])

        x = np.stack([aug(image=i)["image"] for i in x])  # Pixel-level modifications
        return (
            x,
            y,
        )

    def generate_ds(
        self,
        name: str,
        batch: int = 32,
        prefilter: Callable | NoneType = None,
    ):
        cache_dir = (
            "~/.cache/huggingface/datasets/"
            + self.name.lower()
            + "-"
            + name.lower()
            + "/"
        )
        if os.path.exists(os.path.expanduser(cache_dir)):
            ds = load_from_disk(cache_dir)
        else:
            ds = self.fetch_dataset()
            if not prefilter:

                def prefilter(features: list[PIL.Image.Image]):
                    batch = {}
                    batch["image"] = [
                        utils.array_to_img(
                            clahe(
                                utils.img_to_array(f, dtype="uint8"),
                                clip_limit=1.0,
                                tile_grid_size=(8, 8),
                            )
                        )
                        for f in features
                    ]
                    return batch

            ds = ds.map(
                lambda x: prefilter(x),
                input_columns=self.value[-1][-1],
                batched=True,
                batch_size=batch,
                num_proc=cpu_count(),
            )

            if "target" in ds.column_names:
                ds = ds.select_columns(self.value[-1] + ["target"])
                ds = ds.cast_column("target", features.Image())
            else:
                ds = ds.select_columns(self.value[-1])

            ds = ds.cast_column("image", features.Image())
            ds.save_to_disk(cache_dir)
        return ds

    def get_loaders(
        self,
        ds: Dataset,
        dims: tuple[int, int],
        train_size: float | int | None,
        test_size: float | int | None,
        batch: int = 32,
    ):
        ds = ds.train_test_split(test_size=test_size, train_size=train_size)

        if backend.backend() == "tensorflow":
            train_ds, test_ds = self._load_data_tf(ds["train"], ds["test"], batch)
            augmenters = self.get_augmenters(
                {"augs_per_image": 1, "rate": 0.5},
                {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
            )
            train_ds = (
                train_ds.map(
                    lambda x: self.data_augmenter(
                        x, augmenters["geom"], augmenters["aug"]
                    ),
                    num_parallel_calls=-1,
                )
                .repeat()
                .prefetch(-1)
            )
        elif backend.backend() == "torch":
            train_ds, test_ds = self._load_data_torch(
                dims, ds["train"], ds["test"], batch
            )

        return train_ds, test_ds

    def _load_data_torch(
        self, dims: tuple[int, int], train_ds: Dataset, test_ds: Dataset, batch: int
    ):
        from torch.utils.data import DataLoader

        def augmenter(example):
            example = self.data_collater(example, {"w": dims[0], "h": dims[1]})
            augmenters = self.get_augmenters(
                {"augs_per_image": 1, "rate": 0.5},
                {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
            )

            x, y = self.data_augmenter(example, augmenters["geom"], augmenters["aug"])
            return (x, y)
            # return {"image": x, "target": y}

        # train_ds.set_transform(augmenter)

        train_ds.set_format("numpy")

        train_ds = DataLoader(
            train_ds,
            batch,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=augmenter,
            num_workers=int(os.getenv("DL_TRAIN_WORKERS", cpu_count() - 4)),
            pin_memory=True,
        )

        def v_transform(examples):
            examples = self.data_collater(examples, {"w": dims[0], "h": dims[1]})

            if isinstance(examples, list):
                x = np.stack(
                    [ex[self.value[-1][0]] for ex in examples], dtype="float32"
                )
                if len(self.value[-1]) > 1:
                    y = np.stack(
                        [ex[self.value[-1][1]] for ex in examples], dtype="float32"
                    )
                elif "target" in examples[0].keys():
                    y = np.stack([ex["target"] for ex in examples], dtype="float32")
                else:
                    y = np.copy(x)
            else:
                x = np.stack(examples[self.value[-1][0]], dtype="float32")
                # Determine if desired output is a referenced output or the original image
                if len(self.value[-1]) > 1:
                    y = np.stack(examples[self.value[-1][1]], dtype="float32")
                elif "target" in examples.keys():
                    y = np.stack(examples["target"], dtype="float32")
                else:
                    y = np.copy(x)  # Separate out target

            return (x, y)
            # return {"image": x, "target": y}

        test_ds.set_format("numpy")

        test_ds = DataLoader(
            test_ds,
            batch,
            shuffle=False,
            persistent_workers=True,
            collate_fn=v_transform,
            num_workers=int(os.getenv("DL_TEST_WORKERS", 4)),
            pin_memory=True,
        )

        return train_ds, test_ds

    def _load_data_tf(self, train_ds: Dataset, test_ds: Dataset, batch: int):
        train_ds = train_ds.to_tf_dataset(
            columns=self.value[2],
            batch_size=batch,
            collate_fn=self.data_collater,
            # collate_fn_args={"args": {"w": dim[0], "h": dim[1]}},
            prefetch=False,
            shuffle=True,
            drop_remainder=True,
        )

        test_ds = (
            test_ds.to_tf_dataset(
                columns=self.value[2],
                batch_size=batch,
                collate_fn=self.data_collater,
                # collate_fn_args={"args": {"w": dim[0], "h": dim[1]}},
                prefetch=True,
                drop_remainder=True,
            )
            .cache()
            .prefetch(-1)
        )
        return train_ds, test_ds
