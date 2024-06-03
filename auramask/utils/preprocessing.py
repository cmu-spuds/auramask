from keras_cv.layers import (
    RandomRotation,
    Augmenter,
    Equalization,
    Rescaling,
    Resizing,
    RandomAugmentationPipeline,
    RandomFlip,
    RandomTranslation,
    RandAugment,
)
from keras.layers import CenterCrop


# TODO: the w and h refer to the resampled and not center-cropped. Could be misleading to some users.
def gen_image_loading_layers(w: int, h: int, crop: bool = True) -> Augmenter:
    """Generate an image processing augmentation pipeline that converts the image to a [0,1] scale, resizes to w, h and center crops to 224, 224

    Args:
        w (int): Desired width of the images
        h (int): Desired height of the images
        crop (bool, optional): Whether to crop to aspect ratio or not. Defaults to True.

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return Augmenter(
        [
            Equalization((0, 255)),
            Resizing(w, h, crop_to_aspect_ratio=True),
            Rescaling(scale=1.0 / 255, offset=0),
            CenterCrop(224, 224),
        ]
    )


def gen_geometric_aug_layers(augs_per_image: int, rate: float = 10 / 11) -> Augmenter:
    """Generate an image processing augmentation pipeline that applies geometric modifications
    to the input images.

    Args:
        augs_per_image (int): Number of augmentations to apply to an image
        rate (float): The probability that an augmentation will be applied. Defaults to 10/11

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return Augmenter(
        [
            RandomAugmentationPipeline(
                [
                    RandomRotatePairs(factor=0.5),
                    RandomFlip(mode="horizontal_and_vertical"),
                    RandomTranslation(
                        height_factor=0.2, width_factor=0.3, fill_mode="nearest"
                    ),
                ],
                augmentations_per_image=augs_per_image,
                rate=rate,
            ),
        ]
    )


def gen_non_geometric_aug_layers(
    augs_per_image: int, rate: float = 10 / 11, magnitude: float = 0.2
) -> Augmenter:
    """Generate an image processing augmentation pipeline that applies non-geometric modifications
    to the input images.

    Args:
        augs_per_image (int): Number of augmentations to apply to an image
        rate (float, optional): The probability that an augmentation will be applied. Defaults to 10/11.
        magnitude (float, optional): The magnitude of the changes made in range [0,1]. Defaults to 0.2.

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return Augmenter(
        [
            RandAugment(
                value_range=(0, 1),
                augmentations_per_image=augs_per_image,
                magnitude=magnitude,
                geometric=False,
                rate=rate,
            ),
        ]
    )


class RandomRotatePairs(RandomRotation):
    """Custom random rotation class to rotate both the X data and its paired Y data.

    Args:
        RandomRotation (_type_): _description_
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment_segmentation_masks(self, segmentation_masks, transformations, **kwargs):
        return self._rotate_images(segmentation_masks, transformations)
