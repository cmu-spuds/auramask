# """
# Copyright 2021 Mahmoud Afifi.
#  Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
#  Controlling Colors of GAN-Generated and Real Images via Color Histograms."
#  In CVPR, 2021.
#  @inproceedings{afifi2021histogan,
#   title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
#   Color Histograms},
#   author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
#   booktitle={CVPR},
#   year={2021}
# }
# """

# import keras
# from keras import ops

# H = 1
# W = 2
# C = 0

# EPS = 1e-6


# class RGBuvHistBlock(keras.Layer):
#     def __init__(
#         self,
#         h=64,
#         insz=150,
#         resizing="interpolation",
#         method="inverse-quadratic",
#         sigma=0.02,
#         intensity_scale=True,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.h = h
#         self.insz = insz
#         self.resizing = resizing
#         self.method = method
#         self.intensity_scale = intensity_scale
#         if self.method == "thresholding":
#             self.eps = 6.0 / h
#         else:
#             self.sigma = sigma

#     def forward(self, x):
#         x = ops.clip(x, 0, 1)
#         if x.shape[2] > self.insz or x.shape[3] > self.insz:
#             if self.resizing == "interpolation":
#                 x_sampled = ops.image.resize(
#                     x, size=(self.insz, self.insz), interpolation="bilinear"
#                 )
#             elif self.resizing == "sampling":
#                 inds_1 = ops.cast(
#                     ops.linspace(0, x.shape[2], self.h, endpoint=False), dtype="int64"
#                 )
#                 inds_2 = ops.cast(
#                     ops.linspace(0, x.shape[3], self.h, endpoint=False), dtype="int64"
#                 )
#                 x_sampled = x.index_select(2, inds_1)
#                 x_sampled = x_sampled.index_select(3, inds_2)
#             # else:
#             # 	raise Exception(
#             # 		f'Wrong resizing method. It should be: interpolation or sampling. '
#             # 		f'But the given value is {self.resizing}.')
#         else:
#             x_sampled = x

#         L = x_sampled.shape[0]  # size of mini-batch
#         if x_sampled.shape[1] > 3:
#             x_sampled = x_sampled[:, :3, :, :]
#         X = ops.unstack(x_sampled, axis=0)
#         hists = ops.zeros((x_sampled.shape[0], 3, self.h, self.h))
#         for l in range(L):
#             I = torch.t(ops.reshape(X[l], (3, -1)))
#             II = ops.power(I, 2)
#             if self.intensity_scale:
#                 Iy = ops.expand_dims(
#                     ops.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), axis=1
#                 )
#             else:
#                 Iy = 1

#             Iu0 = ops.expand_dims(
#                 ops.log(I[:, 0] + EPS) - ops.log(I[:, 1] + EPS), axis=1
#             )
#             Iv0 = ops.expand_dims(
#                 ops.log(I[:, 0] + EPS) - ops.log(I[:, 2] + EPS), axis=1
#             )
#             diff_u0 = abs(
#                 Iu0
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )
#             diff_v0 = abs(
#                 Iv0
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )
#             if self.method == "thresholding":
#                 diff_u0 = ops.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
#                 diff_v0 = ops.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
#             elif self.method == "RBF":
#                 diff_u0 = (
#                     ops.power(ops.reshape(diff_u0, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v0 = (
#                     ops.power(ops.reshape(diff_v0, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u0 = ops.exp(-diff_u0)  # Radial basis function
#                 diff_v0 = ops.exp(-diff_v0)
#             elif self.method == "inverse-quadratic":
#                 diff_u0 = (
#                     ops.power(ops.reshape(diff_u0, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v0 = (
#                     ops.power(ops.reshape(diff_v0, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
#                 diff_v0 = 1 / (1 + diff_v0)
#             # else:
#             # 	raise Exception(
#             # 		f'Wrong kernel method. It should be either thresholding, RBF,'
#             # 		f' inverse-quadratic. But the given value is {self.method}.')
#             diff_u0 = ops.cast(diff_u0, "float32")
#             diff_v0 = ops.cast(diff_v0, "float32")
#             a = torch.t(Iy * diff_u0)
#             hists[l, 0, :, :] = torch.mm(a, diff_v0)

#             Iu1 = ops.expand_dims(
#                 ops.log(I[:, 1] + EPS) - ops.log(I[:, 0] + EPS), axis=1
#             )
#             Iv1 = ops.expand_dims(
#                 ops.log(I[:, 1] + EPS) - ops.log(I[:, 2] + EPS), axis=1
#             )
#             diff_u1 = abs(
#                 Iu1
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )
#             diff_v1 = abs(
#                 Iv1
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )

#             if self.method == "thresholding":
#                 diff_u1 = ops.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
#                 diff_v1 = ops.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
#             elif self.method == "RBF":
#                 diff_u1 = (
#                     ops.power(ops.reshape(diff_u1, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v1 = (
#                     ops.power(ops.reshape(diff_v1, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u1 = ops.exp(-diff_u1)  # Gaussian
#                 diff_v1 = ops.exp(-diff_v1)
#             elif self.method == "inverse-quadratic":
#                 diff_u1 = (
#                     ops.power(ops.reshape(diff_u1, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v1 = (
#                     ops.power(ops.reshape(diff_v1, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
#                 diff_v1 = 1 / (1 + diff_v1)

#             diff_u1 = ops.cast(diff_u1, "float32")
#             diff_v1 = ops.cast(diff_v1, "float32")
#             a = torch.t(Iy * diff_u1)
#             hists[l, 1, :, :] = torch.mm(a, diff_v1)

#             Iu2 = ops.expand_dims(
#                 ops.log(I[:, 2] + EPS) - ops.log(I[:, 0] + EPS), axis=1
#             )
#             Iv2 = ops.expand_dims(
#                 ops.log(I[:, 2] + EPS) - ops.log(I[:, 1] + EPS), axis=1
#             )
#             diff_u2 = abs(
#                 Iu2
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )
#             diff_v2 = abs(
#                 Iv2
#                 - ops.expand_dims(
#                     ops.cast(ops.linspace(-3, 3, num=self.h), "float32"), axis=0
#                 )
#             )
#             if self.method == "thresholding":
#                 diff_u2 = ops.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
#                 diff_v2 = ops.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
#             elif self.method == "RBF":
#                 diff_u2 = (
#                     ops.power(ops.reshape(diff_u2, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v2 = (
#                     ops.power(ops.reshape(diff_v2, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u2 = ops.exp(-diff_u2)  # Gaussian
#                 diff_v2 = ops.exp(-diff_v2)
#             elif self.method == "inverse-quadratic":
#                 diff_u2 = (
#                     ops.power(ops.reshape(diff_u2, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_v2 = (
#                     ops.power(ops.reshape(diff_v2, (-1, self.h)), 2) / self.sigma**2
#                 )
#                 diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
#                 diff_v2 = 1 / (1 + diff_v2)
#             diff_u2 = ops.cast(diff_u2, "float32")
#             diff_v2 = ops.cast(diff_v2, "float32")
#             a = torch.t(Iy * diff_u2)
#             hists[l, 2, :, :] = torch.mm(a, diff_v2)

#         # normalization
#         hists_normalized = hists / (
#             ((hists.sum(axis=1)).sum(axis=1)).sum(axis=1).view(-1, 1, 1, 1) + EPS
#         )

#         return hists_normalized
