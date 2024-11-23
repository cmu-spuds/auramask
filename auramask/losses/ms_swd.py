from types import NoneType
from keras import Loss, ops, backend as K

from auramask.utils.stylerefs import StyleRefs


class MSSWD(Loss):
    def __init__(
        self,
        reference: StyleRefs | NoneType = StyleRefs.DIM,
        name="MSSWD",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        self.reference = reference
        if reference:
            # Color reference
            self.target_img = reference.get_img()
            self.target_img = ops.image.resize(
                self.target_img, (224, 224), crop_to_aspect_ratio=True
            )
        else:
            self.target_img = None

        import pyiqa

        self.model = pyiqa.create_metric("msswd", as_loss=True)

    def get_config(self):
        return super().get_config()

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        if self.target_img is not None:
            y_true = self.target_img
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return self.model(ref=y_true, target=y_pred)
