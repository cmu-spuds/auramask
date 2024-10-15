from keras import Loss, layers, applications, Model, ops, losses


class ContentLoss(Loss):
    def __init__(
        self,
        name="ContentLoss",
        content_layer: list[str] = "block5_conv2",
        distance: Loss = losses.MeanSquaredError(),
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        global model_obj

        if "model_obj" not in globals():
            model_obj = {}

        if "vgg19" not in model_obj.keys():
            inp = layers.Input(shape=(None, None, 3))
            x = applications.vgg19.preprocess_input(inp)
            model = applications.VGG19(
                weights="imagenet", include_top=False, input_tensor=x
            )
            model.trainable = False
            model_obj["vgg19"] = model

        base = model_obj["vgg19"]

        self.distance = distance

        output = None
        for layer in base.layers:
            if layer.name is content_layer:
                output = layer.output

        # Feature extractor
        self.feature_extractor = Model(inputs=inp, outputs=output)
        self.feature_extractor.trainable = False

    def get_config(self):
        base_config = super().get_config()
        config = {
            "distance": self.distance.name,
        }
        return {**base_config, **config}

    def call(self, X, y_pred):
        y_pred = ops.multiply(y_pred, 255.0)
        X = ops.multiply(X, 255.0)

        X_features = self.feature_extractor(X, training=False)
        pred_features = self.feature_extractor(y_pred, training=False)

        # Add content loss
        loss = self.distance(X_features, pred_features)

        return loss
