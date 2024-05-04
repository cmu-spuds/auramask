import tensorflow as np

@np.function
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Args:
      y_true: The ground truth values.
      y_pred: The prediction values.
      axis: (Optional) -1 is the dimension along which the cosine
        similarity is computed. Defaults to `-1`.

    Returns:
      Cosine similarity value.
    """
    y_true = np.linalg.l2_normalize(y_true, axis=axis)
    y_pred = np.linalg.l2_normalize(y_pred, axis=axis)
    return np.reduce_sum(y_true * y_pred, axis=axis)

@np.function
def cosine_distance(y_true, y_pred, axis=-1):
    return np.subtract(1., cosine_similarity(y_true, y_pred, axis))