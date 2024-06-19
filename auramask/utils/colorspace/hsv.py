from auramask.utils.colorspace.color_conversion import ColorConversion
from keras import KerasTensor, ops, backend
from math import pi


def hsv_to_rgb(X):
    """Helper function to convert from hsv to rgb while passing through the gradient as
    it has not been implmented in a differentiable way.

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    if backend.is_keras_tensor(X):
        return HSVtoRGB(data_format=backend.image_data_format()).symbolic_call(X)
    else:
        return _hsv_to_rgb(X, backend.image_data_format())


def rgb_to_hsv(X):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    if backend.is_keras_tensor(X):
        return RGBtoHSV(data_format=backend.image_data_format()).symbolic_call(X)
    else:
        return _rgb_to_hsv(X, data_format=backend.image_data_format())


class RGBtoHSV(ColorConversion):
    def __init__(self, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, image):
        return _rgb_to_hsv(image, self.eps, self.data_format)


def _rgb_to_hsv(image, eps: float = 1e-6, data_format: str = "channels_last"):
    if not isinstance(image, KerasTensor):
        raise TypeError(f"Input type is not a KerasTensor. Got {type(image)}")

    if data_format == "channels_last":
        c_axis = -1
    else:
        c_axis = -3

    if len(image.shape) < 3 or image.shape[c_axis] != 3:
        if c_axis == -1:
            raise ValueError(
                f"Input size must have a shape of (*, W, H, 3). Got {image.shape}"
            )
        else:
            raise ValueError(
                f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
            )

    max_rgb, argmax_rgb = ops.max(image, c_axis)
    min_rgb, _ = ops.min(image, c_axis)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = ops.divide(deltac, ops.add(max_rgb + eps))

    deltac = ops.where(deltac == 0, ops.ones_like(deltac), deltac)
    rc, gc, bc = ops.unstack((max_rgb.unsqueeze(c_axis) - image), axis=c_axis)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = ops.stack((h1, h2, h3), dim=c_axis) / deltac.unsqueeze(c_axis)
    h = ops.take_along_axis(h, dim=c_axis, index=argmax_rgb.unsqueeze(c_axis)).squeeze(
        c_axis
    )
    h = (h / 6.0) % 1.0
    h = 2.0 * pi * h  # we return 0/2pi output

    return ops.stack((h, s, v), dim=c_axis)


class HSVtoRGB(ColorConversion):
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hsv = RgbToHsv()
        >>> output = hsv(input)  # 2x3x4x5
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, image):
        return _hsv_to_rgb(image, self.data_format)


def _hsv_to_rgb(image, data_format: str = "channels_last"):
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, KerasTensor):
        raise TypeError(f"Input type is not a KerasTensor. Got {type(image)}")

    if data_format == "channels_last":
        c_axis = -1
    else:
        c_axis = -3

    if len(image.shape) < 3 or image.shape[c_axis] != 3:
        if c_axis == -1:
            raise ValueError(
                f"Input size must have a shape of (*, W, H, 3). Got {image.shape}"
            )
        else:
            raise ValueError(
                f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
            )

    if c_axis == -1:
        h: KerasTensor = image[..., :, :, 0] / (2 * pi)
        s: KerasTensor = image[..., :, :, 1]
        v: KerasTensor = image[..., :, :, 2]
    else:
        h: KerasTensor = image[..., 0, :, :] / (2 * pi)
        s: KerasTensor = image[..., 1, :, :]
        v: KerasTensor = image[..., 2, :, :]

    hi: KerasTensor = ops.floor(h * 6) % 6
    f: KerasTensor = ((h * 6) % 6) - hi
    one: KerasTensor = 1.0
    p: KerasTensor = v * (one - s)
    q: KerasTensor = v * (one - f * s)
    t: KerasTensor = v * (one - (one - f) * s)

    hi = ops.cast(hi, "int64")
    indices: KerasTensor = ops.stack([hi, hi + 6, hi + 12], axis=c_axis)
    out = ops.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), axis=c_axis)
    out = ops.take_along_axis(out, indices=indices, axis=c_axis)

    return out
