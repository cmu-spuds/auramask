import argparse
import enum
from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

from random import choice
from string import ascii_uppercase

from auramask.callbacks.callbacks import ImageCallback

from auramask.losses.perceptual import PerceptualLoss
from auramask.losses.embeddistance import EmbeddingDistanceLoss

from auramask.models.face_embeddings import FaceEmbedEnum
from auramask.models.auramask import AuraMask

from git import Repo
branch = Repo('./').active_branch.name

import keras
from keras.layers import CenterCrop
from keras_cv.layers import Resizing, Rescaling, Augmenter, RandAugment
from keras.optimizers import Adam

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
  parser.add_argument('-d', '--depth', type=int, default=5)
  parser.add_argument('-L', '--lpips', type=str, default='alex', choices=['alex', 'vgg', 'squeeze', 'none'])
  parser.add_argument('-F', type=FaceEmbedEnum, nargs="+", required=True, action=EnumAction)
  parser.add_argument('-S', '--seed', type=str, default=''.join(choice(ascii_uppercase) for _ in range(12)))
  parser.add_argument('--log', default=True, type=bool, action=argparse.BooleanOptionalAction)
  parser.add_argument('--t_split', type=str, default='train')
  parser.add_argument('--v_split', type=str, default='test')
  parser.add_argument('--n_filters', type=int, default=64)
  parser.add_argument('--eager', default=False, type=bool, action=argparse.BooleanOptionalAction)
  parser.add_argument('-v', '--verbose', default=1, type=int)
  parser.add_argument('--note', default=True, type=bool, action=argparse.BooleanOptionalAction)
  
  return parser.parse_args()

def load_data():
  (t_ds, v_ds), info = tfds.load('lfw',
                      decoders=tfds.decode.PartialDecoding({
                        'image': True,
                      }),
                      shuffle_files=True,
                      with_info=True,
                      download=True,
                      as_supervised=False,
                      split=[hparams['t_split'], hparams['v_split']])

  return get_data_generator(t_ds, info, hparams['t_split']), get_data_generator(v_ds, info, hparams['v_split'], False)

def get_data_generator(ds, info, split, augment=True):
  loader = Augmenter(
    [
      Rescaling(scale=1./255, offset=0),
      Resizing(hparams['input'][0], hparams['input'][1], crop_to_aspect_ratio=True),
      CenterCrop(224, 224)
    ]
  )
  
  augmenter = Augmenter(
    [
      RandAugment(
        value_range=(0,1),
        augmentations_per_image=3,
        magnitude=0.5,
        seed=hparams['seed']
        )
    ]
  )

  def load_img(images):
    outputs = loader(images)
    return outputs

  def preprocess_data(images, augment=True):
    if augment:
      outputs = augmenter(images)
    else:
      outputs = images
    return outputs, outputs

  t_ds = ds.map(lambda x: load_img(x['image']), num_parallel_calls=AUTOTUNE)
  
  gen_ds = (
    t_ds
    .cache()
    .shuffle(info.splits[split].num_examples)
    .batch(hparams['batch'])
    .map(lambda x: preprocess_data(x, augment))
    .prefetch(buffer_size=AUTOTUNE)
  )  
  return gen_ds

def initialize_loss():
  FLoss = EmbeddingDistanceLoss(F=hparams['F'])
  if hparams['lpips_backbone'] != 'none':
    return [PerceptualLoss(backbone=hparams['lpips_backbone']), FLoss], [hparams['lambda'], 1.]
  else:
    return [FLoss], [1.]
  
def initialize_model():
  model = AuraMask(n_filters=hparams['n_filters'], n_dims=3, eps=hparams['epsilon'], depth=hparams['depth'])

  hparams['model'] = model.model.name
  losses, losses_w = initialize_loss()
  optimizer = Adam(learning_rate=hparams['alpha'])
  model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=losses_w,
    run_eagerly=hparams['eager']
  )
  return model

def set_seed():
  seed = hparams['seed']
  seed = hash(seed) % (2**32)
  keras.utils.set_random_seed(seed)
  hparams['seed'] = seed

def get_sample_data(ds):
  for x, _ in ds.take(1):
    return x


def init_callbacks(sample, logdir, note='', summary=False):
  # histogram_freq = hparams['epochs'] // 10
  tensorboard_callback = ImageCallback(
    sample=sample, 
    log_dir=logdir, 
    update_freq='epoch',
    histogram_freq=100,
    image_frequency=50,
    mask_frequency=25,
    note=note,
    model_summary=summary,
    hparams=hparams
  )
  # early_stop = EarlyStopping(monitor='loss', patience=3)
  return [tensorboard_callback]
  
def main():
  args = parse_args()
  hparams.update(args.__dict__)
  log = hparams.pop('log')
  note = hparams.pop('note')
  verbose = hparams.pop('verbose')
  set_seed()

  t_ds, v_ds = load_data()

  model = initialize_model()

  if log:
    if note:
      note = input("Note for Run:")
    else:
      note = ''
    logdir = 'logs/%s/%s/%s'%(branch, datetime.now().strftime("%m-%d"), datetime.now().strftime("%H.%M"))
    sample = get_sample_data(v_ds)
    model(sample)
    callbacks = init_callbacks(sample, logdir, note, summary=False)
  else:
    callbacks = None

  training_history = model.fit(
    x=t_ds,
    callbacks=callbacks,
    epochs=hparams['epochs'],
    verbose=verbose,
    validation_data=v_ds
  )
  return training_history

if __name__ == "__main__":
  main()
