import tensorflow as tf
from keras.losses import Loss

@tf.function
def tensor2freq(x, patch_factor):
    _, h, w, _ = x.shape

    patch_list = []
    patch_h = h // patch_factor
    patch_w = w // patch_factor
    for i in range(patch_factor):
        for j in range(patch_factor):
            patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

    # stack to patch tensor
    y = tf.stack(patch_list, axis=1)

    # perform 2D DFT (real-to-complex, orthonormalization)
    freq = tf.signal.fft2d(y)
    freq_real = tf.math.real(freq)/freq.shape[-1]
    freq_imag = tf.math.imag(freq)/freq.shape[-1]
    freq = tf.stack([freq_real, freq_imag], axis=-1)
    return freq

class FocalFrequencyLoss(Loss):
    """The keras.losses class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.patch_factor = tf.constant(patch_factor, tf.int32)
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = tf.math.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = tf.math.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / tf.math.reduce_max(matrix_tmp)
            else:
                matrix_tmp = matrix_tmp / tf.math.reduce_max(tf.reduce_max(matrix_tmp, axis=-1), axis=-1)[:, :, :, None, None]

            matrix_tmp = tf.where(tf.math.is_nan(matrix_tmp), tf.zeros_like(matrix_tmp), matrix_tmp)
            weight_matrix = tf.clip_by_value(matrix_tmp, clip_value_min=0.0, clip_value_max=1.0)

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return tf.math.reduce_mean(loss)

    def call(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (tf.Tensor): of shape (N, H, W, C). Predicted tensor.
            target (tf.Tensor): of shape (N, H, W, C). Target tensor.
            matrix (tf.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = tensor2freq(pred, self.patch_factor)
        target_freq = tensor2freq(target, self.patch_factor)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = tf.math.reduce_mean(pred_freq, axis=0, keepdim=True)
            target_freq = tf.math.reduce_mean(target_freq, axis=0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix)