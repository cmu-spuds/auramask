from keras import ops, Loss, KerasTensor, backend, losses


class HistogramMatchingLoss(Loss):
    def __init__(
        self,
        n_bins: int = 256,
        distance: Loss = losses.MeanSquaredError(),
        name="Histogram Matching Loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_bins = n_bins
        self.distance = distance

    def get_config(self):
        base_config = super().get_config()
        config = {"bins": self.n_bins, "distance": self.distance.name}
        return {**base_config, **config}

    def call(self, y_true: KerasTensor, y_pred: KerasTensor):
        # Reshape for histogram computation
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

        # Compute Histograms
        if backend.backend() == "torch":
            from torch import histc

            hist_true = histc(y_true, self.n_bins, min=0.0, max=1.0)
            hist_pred = histc(y_pred, self.n_bins, min=0.0, max=1.0)
        else:
            hist_true = ops.histogram(y_true, self.n_bins, (0.0, 1.0))
            hist_pred = ops.histogram(y_pred, self.n_bins, (0.0, 1.0))

        # Normalize hists
        hist_true = ops.divide(hist_true, ops.sum(hist_true))
        hist_pred = ops.divide(hist_pred, ops.sum(hist_pred))

        # Compute cumulative density functions (CDFs)
        cdf_true = ops.cumsum(hist_true)
        cdf_pred = ops.cumsum(hist_pred)

        # MSE between CDFs
        loss = self.distance(cdf_true, cdf_pred)

        return loss
