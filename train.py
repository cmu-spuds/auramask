import argparse
import enum
import os
from pathlib import Path
from datasets import load_dataset
from random import choice
from string import ascii_uppercase

from auramask.callbacks.callbacks import ImageCallback

from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss
from auramask.losses.aesthetic import AestheticLoss
from auramask.losses.ssim import SSIMLoss

from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.auramask import AuraMask

import keras
from keras.layers import CenterCrop
from keras_cv.layers import (
    Resizing,
    Rescaling,
    Augmenter,
    RandAugment,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomAugmentationPipeline,
)
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import tensorflow as tf

from datetime import datetime

from git import Repo

from auramask.utils.colorspace import ColorSpaceEnum

branch = Repo("./").active_branch.name  # Used for debugging runs

# Global hparams object
hparams: dict = {}


# Path checking and creation if appropriate
def dir_path(path):
    if path:
        path = Path(path)
        try:
            if not path.parent.parent.exists():
                raise FileNotFoundError()
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except FileNotFoundError as e:
            raise argparse.ArgumentTypeError(
                f"The directory {path} cannot have more than 2 missing parents."
            )
        except FileExistsError as e:
            raise argparse.ArgumentTypeError(f"The directory {path} exists as a file")
    return


# Action for enumeration input
class EnumAction(argparse.Action):
    """Action for an enumeration input, maps enumeration type to choices"""

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name.lower() for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        if isinstance(values, str):
            value = self._enum[values.upper()]
        elif isinstance(values, list):
            value = [self._enum[x.upper()] for x in values]
        setattr(namespace, self.dest, value)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AuraMask Training",
        description="A training script for the AuraMask network",
    )
    parser.add_argument("-a", "--alpha", type=float, default=2e-4)
    parser.add_argument("-e", "--epsilon", type=float, default=0.03)
    parser.add_argument("-l", "--lambda", type=float, default=0.0)
    parser.add_argument("-g", "--gamma", type=float, default=0.1)
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-d", "--depth", type=int, default=5)
    parser.add_argument(
        "-L",
        "--lpips",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze", "mse", "ssim", "ssim+mse", "none"],
    )
    parser.add_argument(
        "--aesthetic", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-F", type=FaceEmbedEnum, nargs="+", required=False, action=EnumAction
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=str,
        default="".join(choice(ascii_uppercase) for _ in range(12)),
    )
    parser.add_argument(
        "--log", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--log-dir", default=None, type=dir_path)
    parser.add_argument("--t-split", type=str, default="train")
    parser.add_argument("--v-split", type=str, default="test")
    parser.add_argument("--n-filters", type=int, default=64)
    parser.add_argument(
        "--eager", default=False, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-v", "--verbose", default=1, type=int)
    parser.add_argument(
        "--note", default=True, type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-C", "--color-space", type=ColorSpaceEnum, action=EnumAction, required=False
    )

    return parser.parse_args()


def load_data():
    (t_ds, v_ds) = load_dataset(
        "logasja/lfw",
        "aug",
        split=[hparams["t_split"], hparams["v_split"]],
    )

    t_ds = t_ds.to_tf_dataset(batch_size=hparams["batch"], shuffle=True)

    v_ds = v_ds.to_tf_dataset(batch_size=hparams["batch"])

    return get_data_generator(t_ds, True), get_data_generator(v_ds, False)


def get_data_generator(ds, augment=True):
    loader = Augmenter(
        [
            Rescaling(scale=1.0 / 255, offset=0),
            Resizing(
                hparams["input"][0], hparams["input"][1], crop_to_aspect_ratio=True
            ),
            CenterCrop(224, 224),
        ]
    )

    geom_aug = Augmenter(
        [
            RandomAugmentationPipeline(
                [
                    RandomRotation(factor=0.5),
                    RandomFlip(mode="horizontal_and_vertical"),
                    RandomTranslation(
                        height_factor=0.2, width_factor=0.3, fill_mode="nearest"
                    ),
                ],
                augmentations_per_image=1,
                rate=0.5,
            ),
        ]
    )

    augmenter = Augmenter(
        [
            RandAugment(
                value_range=(0, 1),
                augmentations_per_image=3,
                magnitude=0.5,
                geometric=False,
            ),
        ]
    )

    def load_img(images, augment=True):
        x = loader(images["orig"])
        y = loader(images["aug"])
        if augment:
            data = geom_aug({"images": x, "segmentation_masks": y})
            print(data)
            data = augmenter(data)
            return data["images"], data["segmentation_masks"]
        else:
            return x, y

    t_ds = ds.map(lambda x: load_img(x, augment), num_parallel_calls=-1)

    # print(t_ds)

    return t_ds


def initialize_loss():
    losses = []
    weights = []
    if hparams["F"]:
        losses.append(EmbeddingDistanceLoss(F=hparams["F"]))
        weights.append(1.0)
    if hparams["aesthetic"]:
        losses.append(AestheticLoss(kind="nima"))
        weights.append(hparams["gamma"])
    if hparams["lambda"] > 0:
        if hparams["lpips"] == "none":
            pass
        elif hparams["lpips"] == "mse":
            losses.append(MeanSquaredError())
            weights.append(hparams["lambda"])
        elif hparams["lpips"] == "ssim":
            losses.append(SSIMLoss())
            weights.append(hparams["lambda"])
        elif hparams["lpips"] == "ssim+mse":
            losses.append(SSIMLoss())
            weights.append(hparams["lambda"])
            losses.append(MeanSquaredError())
            weights.append(hparams["lambda"])
        else:
            losses.append(PerceptualLoss(backbone=hparams["lpips"]))
            weights.append(hparams["lambda"])

    return losses, weights


def initialize_model():
    with tf.device("gpu:0"):
        model = AuraMask(
            n_filters=hparams["n_filters"],
            n_dims=3,
            eps=hparams["epsilon"],
            depth=hparams["depth"],
            colorspace=hparams["color_space"].value,
        )

    hparams["model"] = model.model.name

    with tf.device("gpu:0"):
        losses, losses_w = initialize_loss()
    optimizer = Adam(learning_rate=hparams["alpha"])
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=losses_w,
        run_eagerly=hparams["eager"],
    )
    return model


def set_seed():
    seed = hparams["seed"]
    seed = hash(seed) % (2**32)
    keras.utils.set_random_seed(seed)
    hparams["seed"] = seed


def get_sample_data(ds):
    for x, y in ds.take(1):
        inp = tf.identity(x[:5])
        outp = tf.identity(y[:5])
    return tf.stack([inp, outp])


def init_callbacks(sample, logdir, note="", summary=False):
    # histogram_freq = hparams['epochs'] // 10
    tensorboard_callback = ImageCallback(
        sample=sample,
        log_dir=logdir,
        update_freq="epoch",
        histogram_freq=10,
        image_frequency=5,
        mask_frequency=5,
        model_checkpoint_frequency=0,
        note=note,
        model_summary=summary,
        hparams=hparams,
    )
    # early_stop = EarlyStopping(monitor='loss', patience=3)
    return [tensorboard_callback]


def main():
    # Constant Defaults
    hparams["optimizer"] = "adam"
    hparams["input"] = (256, 256)
    hparams.update(parse_args().__dict__)
    # print(hparams)
    log = hparams.pop("log")
    logdir = hparams.pop("log_dir")
    note = hparams.pop("note")
    verbose = hparams.pop("verbose")
    set_seed()

    # Load the training and validation data
    t_ds, v_ds = load_data()

    # Initialize the model with the input hyperparameters
    model = initialize_model()

    if log:
        if note:
            note = input("Note for Run:")
        else:
            note = ""
        if not logdir:
            logdir = os.path.join(
                "logs",
                branch,
                datetime.now().strftime("%m-%d"),
                datetime.now().strftime("%H.%M"),
            )
        else:
            logdir = os.path.join(
                logdir,
                datetime.now().strftime("%m-%d"),
                datetime.now().strftime("%H.%M"),
            )
        v = get_sample_data(v_ds)
        t = get_sample_data(t_ds)
        model(v[0])
        callbacks = init_callbacks(
            {"validation": v, "train": t}, logdir, note, summary=False
        )
    else:
        callbacks = None

    training_history = model.fit(
        x=t_ds,
        callbacks=callbacks,
        epochs=hparams["epochs"],
        verbose=verbose,
        validation_data=v_ds,
    )
    return training_history


if __name__ == "__main__":
    main()
