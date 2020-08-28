import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np

class F1Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, inputs, targets, filename, patience):
    self.model = model
    self.inputs = inputs
    self.targets = targets
    self.filename = filename
    self.patience = patience

    self.best_score = 0
    self.bad_epoch = 0

  def on_epoch_end(self, epoch, logs):
    pred = self.model.predict(self.inputs)
    score = f1_score(self.targets[:,0], np.round(pred[:,0]))

    if score > self.best_score:
      self.best_score = score
      self.model.save_weights(self.filename)
      print ("\nScore {}. Model saved in {}.".format(score, self.filename))
      self.bad_epoch = 0
    else:
      print ("\nScore {}. Model not saved.".format(score))
      self.bad_epoch += 1

    if self.bad_epoch >= self.patience:
      print ("\nEpoch {}: early stopping.".format(epoch))
      self.model.stop_training = True

class Snapshot(tf.keras.callbacks.Callback):
  def __init__(self, model, inputs, targets, test_inputs=None):
    self.model = model
    self.inputs = inputs
    self.targets = targets
    self.test_inputs = test_inputs

    self.best_score = 0

    self.all_snapshots = []
    self.best_scoring_snapshots = []

    self.all_test_snapshots = []
    self.best_scoring_test_snapshots = []

  def on_epoch_end(self, epoch, logs):
    pred = self.model.predict(self.inputs)
    score = f1_score(self.targets[:,0], np.round(pred[:,0]))

    self.all_snapshots.append(pred)
    if (self.test_inputs is None) == False:
      test_pred = self.model.predict(self.test_inputs)
      self.all_test_snapshots.append(test_pred)

    if score > self.best_score:
      self.best_score = score
      self.best_scoring_snapshots.append(pred)
      if (self.test_inputs is None) == False:
        self.best_scoring_test_snapshots.append(test_pred)
        