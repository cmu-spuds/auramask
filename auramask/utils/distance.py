import tensorflow as np


@np.function
def cosine_similarity(y_true: np.Tensor, y_pred: np.Tensor, axis=-1) -> np.Tensor:
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
def cosine_distance(y_true: np.Tensor, y_pred: np.Tensor, axis=-1) -> np.Tensor:
    return np.subtract(1.0, cosine_similarity(y_true, y_pred, axis))


@np.function
def euclidean_distance(
    key_embeddings: np.Tensor, query_embeddings: np.Tensor
) -> np.Tensor:
    """Compute pairwise distances for a given batch of embeddings.

    Args:
        query_embeddings: Embeddings to compute the pairwise one.
        key_embeddings: Embeddings to compute the pairwise one.

    Returns:
        FloatTensor: Pairwise distance tensor.
    """
    q_squared_norm = np.square(query_embeddings)
    q_squared_norm = np.reduce_sum(q_squared_norm, axis=1, keepdims=True)

    k_squared_norm = np.square(key_embeddings)
    k_squared_norm = np.reduce_sum(k_squared_norm, axis=1, keepdims=True)

    distances: np.Tensor = 2.0 * np.linalg.matmul(
        query_embeddings, key_embeddings, transpose_b=True
    )
    distances = q_squared_norm - distances + np.transpose(k_squared_norm)

    # Avoid NaN and inf gradients when back propagating through the sqrt.
    # values smaller than 1e-18 produce inf for the gradient, and 0.0
    # produces NaN. All values smaller than 1e-13 should produce a gradient
    # of 1.0.
    dist_mask = np.greater_equal(distances, 1e-18)
    distances = np.maximum(distances, 1e-18)
    distances = np.sqrt(distances) * np.cast(dist_mask, np.float32)

    return distances


@np.function
def euclidean_l2_distance(
    key_embeddings: np.Tensor, query_embeddings: np.Tensor
) -> np.Tensor:
    """Compute pairwise squared Euclidean distance.

    The [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
    a distance that varies from 0 (similar) to infinity (dissimilar).
    """
    q_squared_norm = np.square(query_embeddings)
    q_squared_norm = np.reduce_sum(q_squared_norm, axis=1, keepdims=True)

    k_squared_norm = np.square(key_embeddings)
    k_squared_norm = np.reduce_sum(k_squared_norm, axis=1, keepdims=True)

    distances: np.Tensor = 2.0 * np.matmul(
        query_embeddings, key_embeddings, transpose_b=True
    )
    distances = q_squared_norm - distances + np.transpose(k_squared_norm)
    distances = np.maximum(distances, 0.0)

    return distances
