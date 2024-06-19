from keras import KerasTensor, Operation


class ColorConversion(Operation):
    def __init__(self, data_format="channels_last") -> None:
        super().__init__()
        self.data_format = data_format

    def _conversion_algorithm(self, image):
        raise NotImplementedError

    def call(self, image):
        return self._conversion_algorithm(image)

    def compute_output_spec(self, image):
        if len(image.shape) not in (3, 4):
            raise ValueError(
                "Invalid image rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

        if len(image.shape) == 3:
            if self.data_format == "channels_last":
                return KerasTensor(image.shape[:-1] + (3,), dtype=image.dtype)
            else:
                return KerasTensor((3,) + image.shape[1:], dtype=image.dtype)
        elif len(image.shape) == 4:
            if self.data_format == "channels_last":
                return KerasTensor(
                    (image.shape[0],) + image.shape[1:-1] + (3,),
                    dtype=image.dtype,
                )
            else:
                return KerasTensor(
                    (
                        image.shape[0],
                        3,
                    )
                    + image.shape[2:],
                    dtype=image.dtype,
                )
