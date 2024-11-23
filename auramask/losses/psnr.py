from keras import Loss, ops, backend as K


class IQAPSNR(Loss):
    def __init__(
        self,
        name="psnr",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("psnr", as_loss=True, eps=K.epsilon())
        print(self.model.lower_better)

    def get_config(self):
        return super().get_config()

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        score = self.model(ref=y_true, target=y_pred)
        score = ops.divide(score, 40.0)
        return ops.subtract(1, score)
