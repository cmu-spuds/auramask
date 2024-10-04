# ruff: noqa: E402
import argparse
import enum
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import keras
from auramask.utils.insta_filter import InstaFilterEnum
from auramask.utils.datasets import DatasetEnum

# Global hparams object
hparams: dict = {}
# Normalize network to use channels last ordering
keras.backend.set_image_data_format("channels_last")


# Path checking and creation if appropriate
def dir_path(path):
    if path:
        path = Path(path)
        try:
            if not path.parent.parent.exists():
                raise FileNotFoundError()
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except FileNotFoundError:
            raise argparse.ArgumentTypeError(
                f"The directory {path} cannot have more than 2 missing parents."
            )
        except FileExistsError:
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
        prog="Dataset Preprocessing",
        description="A script to generate a preprocessed cached dataset for use by the auramask training script",
    )
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument(
        "-D",
        "--dataset",
        default="lfw",
        type=DatasetEnum,
        action=EnumAction,
        required=True,
    )
    parser.add_argument(
        "--instagram-filter", type=InstaFilterEnum, action=EnumAction, required=False
    )

    args = parser.parse_args()

    return args


def main():
    hparams["input"] = (256, 256)
    hparams.update(parse_args().__dict__)

    # Load the dataset
    ds: DatasetEnum = hparams["dataset"]

    insta: InstaFilterEnum = hparams["instagram_filter"]

    ds.generate_ds(
        insta.name,
        hparams["input"],
        batch=hparams["batch"],
        prefilter=insta.filter_transform,
    )


if __name__ == "__main__":
    main()
