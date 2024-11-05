from keras import ops, KerasTensor, utils, backend as K

# from torch import cosine_similarity as t_cosine_similirity, _euclidean_dist, pairwise_distance


def cosine_similarity(y_true: KerasTensor, y_pred: KerasTensor, axis=-1) -> KerasTensor:
    """Computes the cosine similarity between labels and predictions.

    Args:
      y_true: The ground truth values.
      y_pred: The prediction values.
      axis: (Optional) -1 is the dimension along which the cosine
        similarity is computed. Defaults to `-1`.

    Returns:
      Cosine similarity value.
    """
    y_true = ops.convert_to_tensor(y_true, dtype="float32")
    y_pred = ops.convert_to_tensor(y_pred, dtype="float32")
    y_true = utils.normalize(y_true, axis=axis, order=2)
    y_pred = utils.normalize(y_pred, axis=axis, order=2)
    sim = ops.sum(y_true * y_pred, axis=axis)
    return ops.cast(sim, K.floatx())


def cosine_distance(y_true: KerasTensor, y_pred: KerasTensor, axis=-1) -> KerasTensor:
    return ops.subtract(1.0, cosine_similarity(y_true, y_pred, axis))


def euclidean_distance(
    key_embeddings: KerasTensor, query_embeddings: KerasTensor
) -> KerasTensor:
    """Compute pairwise distances for a given batch of embeddings.

    Args:
        query_embeddings: Embeddings to compute the pairwise one.
        key_embeddings: Embeddings to compute the pairwise one.

    Returns:
        FloatTensor: Pairwise distance tensor.
    """
    q_squared_norm = ops.square(query_embeddings)
    q_squared_norm = ops.sum(q_squared_norm, axis=1, keepdims=True)

    k_squared_norm = ops.square(key_embeddings)
    k_squared_norm = ops.sum(k_squared_norm, axis=1, keepdims=True)

    distances: KerasTensor = 2.0 * ops.matmul(
        query_embeddings, ops.transpose(key_embeddings)
    )
    distances = q_squared_norm - distances + ops.transpose(k_squared_norm)

    # Avoid NaN and inf gradients when back propagating through the sqrt.
    # values smaller than 1e-18 produce inf for the gradient, and 0.0
    # produces NaN. All values smaller than 1e-13 should produce a gradient
    # of 1.0.
    dist_mask = ops.greater_equal(distances, 1e-18)
    distances = ops.maximum(distances, 1e-18)
    distances = ops.sqrt(distances) * ops.cast(dist_mask, K.floatx())

    return distances


def euclidean_l2_distance(
    key_embeddings: KerasTensor, query_embeddings: KerasTensor
) -> KerasTensor:
    """Compute pairwise squared Euclidean distance.

    The [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
    a distance that varies from 0 (similar) to infinity (dissimilar).
    """
    q_squared_norm = ops.square(query_embeddings)
    q_squared_norm = ops.sum(q_squared_norm, axis=1, keepdims=True)

    k_squared_norm = ops.square(key_embeddings)
    k_squared_norm = ops.sum(k_squared_norm, axis=1, keepdims=True)

    distances: KerasTensor = 2.0 * ops.matmul(
        query_embeddings, ops.transpose(key_embeddings)
    )
    distances = q_squared_norm - distances + ops.transpose(k_squared_norm)
    distances = ops.maximum(distances, 0.0)

    return distances
