import io
from datetime import datetime
import os

from keras import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'), expand_nested=True, show_trainable=True)
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


class ImageCallback(TensorBoard):
  def __init__(
    self, 
    sample,
    hparams=None,
    note='',
    image_frequency=None,
    mask_frequency=None,
    model_summary=False,
    model_checkpoint_frequency=None,
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
    **kwargs):
    super().__init__(log_dir=log_dir,
                     histogram_freq=histogram_freq,
                     write_graph=write_graph,
                     write_images=write_images,
                     write_steps_per_second=write_steps_per_second,
                     update_freq=update_freq,
                     profile_batch=profile_batch,
                     embeddings_freq=embeddings_freq,
                     embeddings_metadata=embeddings_metadata,
                     **kwargs
                     )
    self._should_have_vandt = isinstance(sample, tuple)
    if self._should_have_vandt:
      self.sample = tf.concat(sample, axis=0)
    else:
      self.sample = sample
    self.note = note
    self.mask_frequency = mask_frequency
    self.model_summary = model_summary
    self.model_checkpoint_frequency=model_checkpoint_frequency
    self.image_frequency = image_frequency
    self.hparams = hparams
    self._should_write_loss_model_weight = histogram_freq > 0
    
  def on_train_begin(self, logs=None):
    super().on_train_begin(logs)
    with self._train_writer.as_default():
      if self.hparams:
        tmp_hparams = self.hparams
        if tmp_hparams['F']: tmp_hparams['F'] = ",".join(tmp_hparams['F'])
        else: tmp_hparams['F'] = ''
        tmp_hparams['input'] = str(tmp_hparams['input'])
      if os.getenv('SLURM_JOB_NAME') and os.getenv('SLURM_ARRAY_TASK_ID'):
        hp.hparams(tmp_hparams, trial_id='%s-%s'%(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_ARRAY_TASK_ID']))
      else:
        hp.hparams(tmp_hparams)
      if not (self.note == ''):
        tf.summary.text("Run Note", self.note, 0)
      if self.model_summary:
        tf.summary.text("Model Structure", get_model_summary(self.model), 0)

  # def _push_writer(self, writer, step):
  #   """Sets the default writer for custom batch-level summaries."""
  #   if self.update_freq == "epoch":
  #       return

  #   should_record = lambda: tf.equal((step % self.update_freq) * 
  #                                    (step % self.image_frequency) * 
  #                                    (step % self.mask_frequency), 0)
  #   # TODO(b/151339474): Fix deadlock when not using .value() here.
  #   summary_context = (
  #       writer.as_default(step.value()),
  #       tf.summary.record_if(should_record),
  #   )
  #   self._prev_summary_state.append(summary_context)
  #   summary_context[0].__enter__()
  #   summary_context[1].__enter__()    

  # def on_train_batch_end(self, batch, logs=None):
  #   super().on_train_batch_end(batch, logs)
  #   if self.image_frequency and batch % self.image_frequency == 0:
  #     y, mask = self.model(self.sample)
  #     tf.summary.image("Augmented", y, max_outputs=1, step=batch)
  #     tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=1, step=batch)
  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch, logs)
    _should_checkpoint = self.model_checkpoint_frequency and epoch > 0 and (epoch) % self.model_checkpoint_frequency == 0
    if _should_checkpoint:
      self.model.save_weights(os.path.join(self.log_dir, '%04d-model.weights.h5'%(epoch)), overwrite=True)
    with self._train_writer.as_default():
      with tf.name_scope('Epoch'):
        _should_update_img = self.image_frequency and (epoch) % self.image_frequency == 0
        _should_update_mask = self.mask_frequency and (epoch) % self.mask_frequency == 0
        if _should_update_img or _should_update_mask:
          y, mask = self.model(self.sample)
          if self._should_have_vandt:
            if _should_update_img:
              tf.summary.image("Augmented", y, max_outputs=2, step=epoch)
              tf.summary.image("TAugmented", y[-4:], max_outputs=2, step=epoch)
            if _should_update_mask:
              tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=2, step=epoch)            
              tf.summary.image("TMask", (mask[-4:] * 0.5) + 0.5, max_outputs=2, step=epoch)            
          else:
            if _should_update_img: tf.summary.image("Augmented", y, max_outputs=2, step=epoch)
            if _should_update_mask: tf.summary.image("Mask", (mask * 0.5) + 0.5, max_outputs=2, step=epoch)
    
  
  def on_train_end(self, logs=None):
    self.model.save(os.path.join(self.log_dir, 'model.keras'))
    return super().on_train_end(logs)
  
  def _log_epoch_metrics(self, epoch, logs):
    with tf.name_scope('Epoch'):
      super()._log_epoch_metrics(epoch, logs)
            
  def _log_weights(self, epoch):
    """Logs the weights of the Model to TensorBoard."""
    with self._train_writer.as_default():
        with tf.summary.record_if(True):
            for layer in self.model.layers:
              if layer is self.model.model:
                prefix=layer.name + '/'
              elif isinstance(layer, Model):
                prefix=layer.name + '/'
                continue
                # if not self._should_write_loss_model_weight: continue
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
            self._should_write_loss_model_weight = False
            self._train_writer.flush()
