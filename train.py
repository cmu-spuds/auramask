import argparse

from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss

from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.auramask import AuraMask

from git import Repo
branch = Repo('./').active_branch.name

import keras
from keras_cv.layers import Resizing, Rescaling, Augmenter
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, Callback

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from datetime import datetime

def createParser():
  parser = argparse.ArgumentParser(
    prog="AuraMask Training",
    description="A training script for the AuraMask network",
  )
  parser.add_argument('-a', '--alpha', type=float, default=2e-4)
  parser.add_argument('-e', '--epsilon', type=float, default=0.03)
  parser.add_argument('-l', '--lambda', type=float, default=0.)
  parser.add_argument('-B', '--batch_size', type=int, default=32)
  parser.add_argument('-E', '--epochs', type=int, default=5)
  parser.add_argument('--lpips_backbone', type=str, choices=['alex', 'vgg', 'squeeze'])
  parser.add_argument('-F', type=str, nargs="+", choices=['vggface', 'facenet', 'facenet512', 'openface', 'deepface', 'deepid', 'arcface', 'sface'])
  parser.add_argument('-S', '--seed', type=str, default=None)
  parser.add_argument('--note', type=str, )

def load_data():
  ds, info = tfds.load('lfw',
                      decoders=tfds.decode.PartialDecoding({
                        'image': True,
                      }),
                      with_info=True,
                      download=True,
                      as_supervised=False,
                      split='train[0:64]')

  return ds, info

hparams = {
  "alpha": 2e-4,
  "epsilon": 0.03,
  "lambda": 0.,
  "batch": 32,
  "optimizer": "adam",
  # EPOCH = 500  # ReFace training
  "epoch": 500,
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
  
  train_ds = ds.batch(hparams['batch']).map(
    lambda x: preprocess_data(x['image']),
    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
  
  return train_ds

def initialize_loss():
  FLoss = EmbeddingDistanceLoss(F=hparams['F'])
  Lpips = PerceptualLoss(backbone=hparams['Lpips_backbone'])
  return FLoss, Lpips
  
def initialize_model(n_filters=32, eager=False):
  model = AuraMask(n_filters=n_filters, n_dims=3, eps=hparams['epsilon'])
  FLoss, Lpips = initialize_loss()
  optimizer = Adam(learning_rate=hparams['alpha'])
  model.compile(
    optimizer=optimizer,
    loss=[Lpips, FLoss],
    loss_weights=[hparams['lambda'],1.],
    run_eagerly=eager
  )
  return model

def get_sample_data(ds, seed=None):
  for x, _ in ds.take(1):
    return x

class ImageCallback(TensorBoard):
  def __init__(
    self, 
    sample,
    note='',
    **kwargs):
    super().__init__(**kwargs)
    self.sample = sample
    self.epoch = 0
    self.note = note
    
  def on_train_begin(self, logs=None):
    super().on_train_begin(logs)
    tmp_hparams = hparams
    tmp_hparams['F'] = ",".join(tmp_hparams['F'])
    tmp_hparams['input'] = str(tmp_hparams['input'])
    with self._train_writer.as_default():
      hp.hparams(tmp_hparams)
      if not (self.note == ''):
        tf.summary.text("Run Note", self.note)
    
  def on_train_batch_end(self, batch, logs=None):
    with tf.name_scope('E%d-Batch'%self.epoch):
      super().on_train_batch_end(batch, logs)
      if batch % self.update_freq == 0:
        y, mask = self.model(self.sample)
        with self._train_writer.as_default():
          tf.summary.image("Augmented", y, max_outputs=1, step=batch)
          tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=1, step=batch)
          
  def on_epoch_end(self, epoch, logs=None):
      super().on_epoch_end(epoch, logs)
      y, mask = self.model(self.sample)
      with self._train_writer.as_default():
        with tf.name_scope('Epoch'):
          tf.summary.image("Augmented", y, max_outputs=10, step=epoch)
          tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=10, step=epoch)
      self.epoch += 1
      
  def _log_weights(self, epoch):
      """Logs the weights of the Model to TensorBoard."""
      with self._train_writer.as_default():
          with tf.summary.record_if(True):
              for layer in self.model.layers:
                if isinstance(layer, keras.Model):
                  prefix=layer.name + '/'
                else:
                  prefix=''
                for weight in layer.weights:
                    weight_name = weight.name.replace(":", "_")
                    # Add a suffix to prevent summary tag name collision.
                    histogram_weight_name = "%s%s/histogram"%(prefix, weight_name)
                    tf.summary.histogram(
                        histogram_weight_name, weight, step=epoch
                    )
                    if self.write_images:
                        # Add a suffix to prevent summary tag name
                        # collision.
                        image_weight_name = "%s%s/image"%(prefix, weight_name)
                        self._log_weight_as_image(
                            weight, image_weight_name, epoch
                        )
              self._train_writer.flush()

def init_callbacks(sample, logdir, note=''):
  tensorboard_callback = ImageCallback(
    sample=sample, 
    log_dir=logdir, 
    write_graph=True, 
    update_freq=1, 
    histogram_freq=10, 
    note=note,
    # profile_batch=2,
  )
  # early_stop = EarlyStopping(monitor='loss', patience=3)
  return [tensorboard_callback]
  
def run(model, x, callbacks=None, verbosity=0, steps=None):
  training_history = model.fit(
    x=x,
    batch_size=hparams['batch'],
    callbacks=callbacks,
    epochs=hparams['epoch'],
    verbose=verbosity,
    steps_per_epoch=steps,
  )
  return training_history

def main():
  note = input("Note for Run:")
  callbacks = None
  logdir = 'logs/%s/nocrop/%s/%s'%(branch, datetime.now().strftime("%m-%d"), datetime.now().strftime("%H.%M"))
  ds, info = load_data()
  t_ds = get_training_data(ds)
  model = initialize_model(n_filters=8, eager=False)
  sample = get_sample_data(t_ds)
  model(sample)
  # model.summary(expand_nested=True, show_trainable=True)
  callbacks = init_callbacks(sample, logdir, note)
  history = run(model, t_ds, callbacks, 0, None)

if __name__ == "__main__":
  main()
