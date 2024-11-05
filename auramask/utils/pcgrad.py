from typing import Optional
from keras import ops, random, backend as K, KerasTensor
import numpy as np


def __compute_gradients(ys, xs) -> list[KerasTensor]:
    """_summary_

    Args:
        ys (_type_): _description_
        xs (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        list[KerasTensor]: _description_
    """
    if K.backend() == "torch":
        for v in xs:
            v.value.grad = None
        ys.backward(retain_graph=True)
        grads = [v.value.grad for v in xs]
        return grads
    elif K.backend() == "tensorflow":
        from tensorflow import gradients

        return gradients(ys, xs)
    else:
        raise NotImplementedError()


def compute_pc_grads(loss: list, var_list: Optional[list] = None):
    """_summary_

    Args:
        loss (list): _description_
        var_list (Optional[list], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    num_tasks = len(loss)
    loss = ops.stack(loss)
    loss = random.shuffle(loss)

    # Compute gradients for each task
    grads_task = ops.map(
        lambda lss: ops.concatenate(
            [
                ops.reshape(grad, [-1])
                for grad in __compute_gradients(lss, var_list)
                if grad is not None
            ],
            axis=0,
        ),
        loss,
    )

    # Compute gradient projections
    def proj_grad(grad_task):
        for k in range(num_tasks):
            dot_product = ops.dot(grad_task, grads_task[k])
            grad_task = grad_task - ops.minimum(dot_product, 0.0) * grads_task[k]
        return grad_task

    proj_grads_flatten = ops.vectorized_map(proj_grad, grads_task)

    # Unpack flattened projected gradients back to original shape
    proj_grads = []
    for j in range(num_tasks):
        start_idx = 0
        for idx, var in enumerate(var_list):
            grad_shape = ops.shape(var)
            flatten_dim = int(np.prod(grad_shape))
            proj_grad = proj_grads_flatten[j][start_idx : start_idx + flatten_dim]
            proj_grad = ops.copy(ops.reshape(proj_grad, grad_shape))
            if len(proj_grads) < len(var_list):
                proj_grads.append(proj_grad)
            else:
                proj_grads[idx] += proj_grad
            start_idx += flatten_dim
    return proj_grads, var_list
