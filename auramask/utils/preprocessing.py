import numpy as np
import albumentations as A


# TODO: the w and h refer to the resampled and not center-cropped. Could be misleading to some users.
def gen_image_loading_layers(w: int, h: int):
    """Generate an image processing augmentation pipeline that converts the image to w, h and center crops to 224, 224

    Args:
        w (int): Desired width of the images
        h (int): Desired height of the images
        crop (bool, optional): Whether to crop to aspect ratio or not. Defaults to True.

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return A.Compose(
        [
            A.FancyPCA(p=1.0),
            A.ToFloat(max_value=255, p=1),
            A.LongestMaxSize(np.maximum(h, w)),
            A.CenterCrop(int(h * 0.875), int(w * 0.875)),
        ]
    )


def gen_geometric_aug_layers(augs_per_image: int, rate: float = 10 / 11):
    """Generate an image processing augmentation pipeline that applies geometric modifications
    to the input images.

    Args:
        augs_per_image (int): Number of augmentations to apply to an image
        rate (float): The probability that an augmentation will be applied. Defaults to 10/11

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return A.Compose(
        [
            A.SomeOf(
                [
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                ],
                n=augs_per_image,
                p=rate,
            )
        ]
    )


def gen_non_geometric_aug_layers(
    augs_per_image: int, rate: float = 10 / 11, magnitude: float = 0.2
):
    """Generate an image processing augmentation pipeline that applies non-geometric modifications
    to the input images.

    Args:
        augs_per_image (int): Number of augmentations to apply to an image
        rate (float, optional): The probability that an augmentation will be applied. Defaults to 10/11.
        magnitude (float, optional): The magnitude of the changes made in range [0,1]. Defaults to 0.2.

    Returns:
        Augmenter: keras_cv.layers.Augmenter
    """
    return A.Compose(
        [
            A.SomeOf(
                [
                    # A.ColorJitter(),
                    A.GaussianBlur(),
                    A.GaussNoise(std_range=(0.05, 0.1)),
                    A.Sharpen(),
                ],
                n=augs_per_image,
                p=rate,
            )
        ]
    )
