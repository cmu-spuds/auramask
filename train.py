from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss

from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.auramask import AuraMask

from git import Repo
branch = Repo('./').active_branch.name

from keras_cv.layers import Resizing, Rescaling, Augmenter
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, Callback

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from datetime import datetime


def load_data():
  ds, info = tfds.load('lfw',
                      decoders=tfds.decode.PartialDecoding({
                        'image': True,
                      }),
                      with_info=True,
                      download=True,
                      as_supervised=False)

  return ds, info

hparams = {
  "alpha": 2e-4,
  "epsilon": 0.03,
  "lambda": 0.,
  "batch": 32,
  "optimizer": "adam",
  # EPOCH = 500  # ReFace training
  "epoch": 2,
  "F": [FaceEmbedEnum.ARCFACE],
  "Lpips_backbone": "alex",
  "input": (256,256)
}

def get_training_data(ds):
  augmenter = Augmenter(
    [
      Rescaling(1./255),
      Resizing(hparams['input'][0],hparams['input'][1]),
    ]
  )

  def preprocess_data(images, augment=True):
    inputs = {"images": images}
    outputs = augmenter(inputs)
    return outputs['images'], outputs['images']  
  
  train_ds = ds['train'].batch(hparams['batch']).map(
    lambda x: preprocess_data(x['image']),
    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
  
  return train_ds

def initialize_loss():
  FLoss = EmbeddingDistanceLoss(F=hparams['F'])
  Lpips = PerceptualLoss(backbone=hparams['Lpips_backbone'])
  return (FLoss, Lpips)
  
def initialize_model():
  model = AuraMask(n_filters=32, n_dims=3, eps=hparams['epsilon'])
  FLoss, Lpips = initialize_loss()
  optimizer = Adam(learning_rate=hparams['alpha'])
  model.compile(
    optimizer=optimizer,
    loss=[Lpips, FLoss],
    loss_weights=[hparams['lambda'],1.],
    run_eagerly=False
  )
  return model

def get_sample_data(ds, seed=None):
  for x, _ in ds.take(1):
    return x

class ImageCallback(Callback):
  def __init__(self, sample):
    self.epoch = 0
    self.sample = sample
    super().__init__()
  def on_train_begin(self, logs=None):
    tmp_hparams = hparams
    tmp_hparams['F'] = ",".join(tmp_hparams['F'])
    tmp_hparams['input'] = str(tmp_hparams['input'])
    hp.hparams(tmp_hparams)
    tf.summary.image("Original", self.sample, max_outputs=10, step=0)
    return super().on_train_begin(logs)
  def on_train_batch_end(self, batch, logs=None):
    if batch % 20 == 0:
      y, mask = self.model(self.sample)
      tf.summary.image("Augmented/%d"%self.epoch, y, max_outputs=1, step=batch)
      tf.summary.image("Mask/%d"%self.epoch, (mask * 0.5) + 0.5, max_outputs=1, step=batch)
    return super().on_train_batch_begin(batch, logs)
  def on_epoch_end(self, epoch, logs=None):
    y, mask = self.model(self.sample)
    tf.summary.image("Augmented/epoch", y, max_outputs=10, step=epoch)
    tf.summary.image("Mask/epoch", (mask * 0.5) + 0.5, max_outputs=10, step=epoch)
    self.epoch+=1
    return super().on_epoch_end(epoch, logs)

def init_callbacks(sample, logdir):
  tensorboard_callback = TensorBoard(log_dir=logdir, write_images=True, update_freq=1, histogram_freq=1)
  early_stop = EarlyStopping(monitor='loss', patience=3)
  img_call = ImageCallback(sample)
  return [tensorboard_callback, early_stop, img_call]
  
def run(model, x, callbacks=None, verbosity=0):
  training_history = model.fit(
    x=x,
    batch_size=hparams['batch'],
    callbacks=callbacks,
    epochs=hparams['epoch'],
    verbose=verbosity
  )
  return training_history

def main():
  callbacks = None
  logdir = 'logs/nocrop/%s/%s/%s'%(branch, datetime.now().strftime("%Y%m%d"), datetime.now().strftime("%H%M%S"))
  ds, info = load_data()
  t_ds = get_training_data(ds)
  model = initialize_model()
  callbacks = init_callbacks(get_sample_data(t_ds), logdir)
  history = run(model, t_ds, callbacks, 0)

if __name__ == "__main__":
  main()
