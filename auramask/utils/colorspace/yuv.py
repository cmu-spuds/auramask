from auramask.utils.colorspace.color_conversion import ColorConversion
from keras import ops, backend


def yuv_to_rgb(X, convert: bool = False):
    """Helper function to convert from yuv to rgb while passing through the gradient as
    it has not been implmented in a differentiable way.

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    if backend.is_keras_tensor(X):
        yuv = YUVtoRGB(data_format=backend.image_data_format()).symbolic_call(X)
    else:
        yuv = _yuv_to_rgb(X, backend.image_data_format())
    if convert:
        yuv = ops.add(yuv, [0.0, 0.5, 0.5])
        yuv = ops.clip(yuv, 0.0, 1.0)

    return yuv


def rgb_to_yuv(X, convert: bool = False):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    if convert:
        X = ops.subtract(X, [0.0, 0.5, 0.5])

    if backend.is_keras_tensor(X):
        return RGBtoYUV(data_format=backend.image_data_format()).symbolic_call(X)
    else:
        return _rgb_to_yuv(X, data_format=backend.image_data_format())


class RGBtoYUV(ColorConversion):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, image):
        return _rgb_to_yuv(image, self.data_format)


def _rgb_to_yuv(image, data_format: str = "channels_last"):
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    ndims = ops.ndim(image)

    if data_format == "channels_last":
        c_axis = ndims - 1
    else:
        c_axis = ndims - 3

    if len(image.shape) < 3 or image.shape[c_axis] != 3:
        if c_axis == -1:
            raise ValueError(
                f"Input size must have a shape of (*, W, H, 3). Got {image.shape}"
            )
        else:
            raise ValueError(
                f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
            )

    _rgb_to_yuv_kernel = ops.convert_to_tensor(
        [
            [0.29900, -0.147108, 0.614777],
            [0.58700, -0.288804, -0.514799],
            [0.11400, 0.435912, -0.099978],
        ],
        dtype=image.dtype,
    )

    yuv = ops.tensordot(image, _rgb_to_yuv_kernel, axes=[[c_axis], [0]])

    y, u, v = ops.split(yuv, 3, axis=c_axis)

    y = ops.clip(y, 0.0, 1.0)
    u = ops.clip(u, -0.5, 0.5)
    v = ops.clip(v, -0.5, 0.5)

    return ops.concatenate([y, u, v], axis=c_axis)


class YUVtoRGB(ColorConversion):
    r"""Convert an image from RGB to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        eps: scalar to enforce numarical stability.

    Returns:
        YUV version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> yuv = RgbToHsv()
        >>> output = yuv(input)  # 2x3x4x5
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, image):
        return _yuv_to_rgb(image, self.data_format)


def _yuv_to_rgb(image, data_format: str = "channels_last"):
    r"""Convert an image from YUV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: YUV Image to be converted to YUV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    ndims = len(ops.shape(image))

    if data_format == "channels_last":
        c_axis = ndims - 1
    else:
        c_axis = ndims - 3

    if len(image.shape) < 3 or image.shape[c_axis] != 3:
        if c_axis == -1:
            raise ValueError(
                f"Input size must have a shape of (*, W, H, 3). Got {image.shape}"
            )
        else:
            raise ValueError(
                f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
            )

    _yuv_to_rgb_kernel = ops.convert_to_tensor(
        [[1, 1, 1], [0, -0.394642334, 2.03206185], [1.13988303, -0.58062185, 0]]
    )

    rgb = ops.tensordot(image, _yuv_to_rgb_kernel, axes=[[c_axis], [0]])

    rgb = ops.clip(rgb, 0.0, 1.0)

    return rgb
