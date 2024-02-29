import argparse
import enum

from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

from random import choice
from string import ascii_uppercase
import numpy as np

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

hparams: dict = {
  "alpha": 2e-4,
  "epsilon": 0.03,
  "lambda": 0.,
  "batch": 32,
  "optimizer": "adam",
  "epochs": 500,
  "F": [FaceEmbedEnum.ARCFACE],
  "lpips_backbone": "alex",
  "input": (256,256)
}

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name.lower() for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        if isinstance(values, str):
          value = self._enum[values.upper()]
        elif isinstance(values, list):
          value = [self._enum[x.upper()] for x in values]
        setattr(namespace, self.dest, value)

def parse_args():
  parser = argparse.ArgumentParser(
    prog="AuraMask Training",
    description="A training script for the AuraMask network",
  )
  parser.add_argument('-a', '--alpha', type=float, default=2e-4)
  parser.add_argument('-e', '--epsilon', type=float, default=0.03)
  parser.add_argument('-l', '--lambda', type=float, default=0.)
  parser.add_argument('-B', '--batch_size', dest='batch', type=int, default=32)
  parser.add_argument('-E', '--epochs', type=int, default=5)
  parser.add_argument('-L', '--lpips_backbone', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'])
  parser.add_argument('-F', type=FaceEmbedEnum, nargs="+", required=True, action=EnumAction)
  parser.add_argument('-S', '--seed', type=str, default=''.join(choice(ascii_uppercase) for _ in range(12)))
  parser.add_argument('--log', default=True, type=bool, action=argparse.BooleanOptionalAction)
  parser.add_argument('--split', type=str, default='train')
  parser.add_argument('--n_filters', type=int, default=64)
  parser.add_argument('--eager', default=False, type=bool, action=argparse.BooleanOptionalAction)
  parser.add_argument('-v', '--verbose', default=1, type=int)
  
  return parser.parse_args()

def load_data():
  ds, info = tfds.load('lfw',
                      decoders=tfds.decode.PartialDecoding({
                        'image': True,
                      }),
                      shuffle_files=True,
                      with_info=True,
                      download=True,
                      as_supervised=False,
                      split=hparams['split'])

  return get_training_data(ds, info)

def get_training_data(ds, info):
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
  
  train_ds = ds.map(
    lambda x: preprocess_data(x['image']),
    num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.cache()
  train_ds = train_ds.shuffle(info.splits[hparams['split']].num_examples)
  train_ds = train_ds.batch(hparams['batch'])
  train_ds = train_ds.prefetch(AUTOTUNE)
  
  return train_ds

def initialize_loss():
  FLoss = EmbeddingDistanceLoss(F=hparams['F'])
  Lpips = PerceptualLoss(backbone=hparams['lpips_backbone'])
  return FLoss, Lpips
  
def initialize_model():
  model = AuraMask(n_filters=hparams['n_filters'], n_dims=3, eps=hparams['epsilon'])
  FLoss, Lpips = initialize_loss()
  optimizer = Adam(learning_rate=hparams['alpha'])
  model.compile(
    optimizer=optimizer,
    loss=[Lpips, FLoss],
    loss_weights=[hparams['lambda'],1.],
    run_eagerly=hparams['eager']
  )
  return model

def set_seed():
  seed = hparams['seed']
  seed = hash(seed) % (2**32)
  keras.utils.set_random_seed(seed)

def get_sample_data(ds):
  for x, _ in ds.take(1):
    return x

class ImageCallback(TensorBoard):
  def __init__(
    self, 
    sample,
    note='',
    mask_frequency = None,
    **kwargs):
    super().__init__(**kwargs)
    self.sample = sample
    self.note = note
    self.mask_frequency = mask_frequency
    
  def on_train_begin(self, logs=None):
    super().on_train_begin(logs)
    tmp_hparams = hparams
    tmp_hparams['F'] = ",".join(tmp_hparams['F'])
    tmp_hparams['input'] = str(tmp_hparams['input'])
    with self._train_writer.as_default():
      hp.hparams(tmp_hparams, trial_id='%s-%s-%s'%(branch, datetime.now().strftime("%m-%d"), datetime.now().strftime("%H.%M")))
      if not (self.note == ''):
        tf.summary.text("Run Note", self.note)
    
  def on_train_batch_end(self, batch, logs=None):
    with tf.name_scope('Batch'):
      super().on_train_batch_end(batch, logs)
      if not isinstance(self.update_freq, str) and batch % self.update_freq == 0:
        y, mask = self.model(self.sample)
        with self._train_writer.as_default():
          tf.summary.image("Augmented", y, max_outputs=1, step=batch)
          tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=1, step=batch)
          
  def on_epoch_end(self, epoch, logs=None):
      super().on_epoch_end(epoch, logs)
      if self.histogram_freq and (epoch+1) % self.histogram_freq == 0:
        y, mask = self.model(self.sample)
        with self._train_writer.as_default():
          with tf.name_scope('Epoch'):
            tf.summary.image("Augmented", y, max_outputs=2, step=epoch)
            tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=2, step=epoch)
            
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
  # histogram_freq = hparams['epochs'] // 10
  tensorboard_callback = ImageCallback(
    sample=sample, 
    log_dir=logdir, 
    update_freq=1, 
    histogram_freq=100,
    mask_frequency=25,
    note=note,
  )
  # early_stop = EarlyStopping(monitor='loss', patience=3)
  return [tensorboard_callback]
  
def main():
  args = parse_args()
  hparams.update(args.__dict__)
  log = hparams.pop('log')
  verbose = hparams.pop('verbose')
  set_seed()

  t_ds = load_data()
  model = initialize_model()

  if log:
    note = input("Note for Run:")
    logdir = 'logs/%s/%s/%s'%(branch, datetime.now().strftime("%m-%d"), datetime.now().strftime("%H.%M"))
    sample = get_sample_data(t_ds)
    model(sample)
    callbacks = init_callbacks(sample, logdir, note)
  else:
    callbacks = None
  # model.summary(expand_nested=True, show_trainable=True)
  training_history = model.fit(
    x=t_ds,
    batch_size=hparams['batch'],
    callbacks=callbacks,
    epochs=hparams['epochs'],
    verbose=verbose,
  )
  return training_history

if __name__ == "__main__":
  main()
