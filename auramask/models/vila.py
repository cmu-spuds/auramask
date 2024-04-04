import tensorflow_hub as hub
from keras.models import Model


class VILA(Model):
    """_summary_

    Args:
    """

    def __init__(self, name="VILA", **kwargs):
        super().__init__(name=name, **kwargs)

        self.net = hub.KerasLayer(
            "https://tfhub.dev/google/vila/image/1",
            signature="serving_default",
            output_key="predictions",
        )

    def get_config(self):
        return self().get_config()

    def call(self, x):
        return self.net(x)
