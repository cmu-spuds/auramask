from enum import Enum
import pilgram2 as pilgram
from PIL.Image import Image
from albumentations import clahe
from keras import utils


class InstaFilterEnum(Enum):
    _1977 = 0
    ADEN = 1
    BRANNAN = 2
    BROOKLYN = 3
    CLARENDON = 4
    EARLYBIRD = 5
    GINGHAM = 6
    HUDSON = 7
    INKWELL = 8
    KELVIN = 9
    LARK = 10
    LOFI = 11
    MAVEN = 12
    MAYFAIR = 13
    MOON = 14
    NASHVILLE = 15
    PERPETUA = 16
    REYES = 17
    RISE = 18
    SLUMBER = 19
    STINSON = 20
    TOASTER = 21
    VALENCIA = 22
    WALDEN = 23
    WILLOW = 24
    XPRO2 = 25

    def filter_transform(self, features: list[Image]):
        batch = {}
        fn = getattr(pilgram, self.name.lower())
        batch["image"] = [
            utils.array_to_img(
                clahe(
                    utils.img_to_array(f, dtype="uint8"),
                    clip_limit=1.0,
                    tile_grid_size=(8, 8),
                )
            )
            for f in features
            # autocontrast(f, preserve_tone=True) for f in features
        ]
        batch["target"] = [fn(f) for f in batch["image"]]
        return batch
