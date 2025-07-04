# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for defining training algorithms."""

import abc
from typing import Dict, List, Optional, Union
from ml_collections.config_dict.config_dict import ConfigDict

import torch
import torch.nn as nn
import numpy as np

from xirl.models import SelfSupervisedOutput

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class Trainer(abc.ABC):
  """Base trainer abstraction.

  Subclasses should override `compute_loss`.
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    """Constructor.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      device: The computing device.
      config: The config dict.
    """
    self._model = model
    self._optimizer = optimizer
    self._device = device
    self._config = config

    self._model.to(self._device).train()
    # self._model.train()

  @abc.abstractmethod
  def compute_loss(
      self,
      embs,
      batch,
  ):
    """Compute the loss on a single batch.

    Args:
      embs: The output of the embedding network.
      batch: The output of a VideoDataset dataloader.

    Returns:
      A tensor corresponding to the value of the loss function evaluated
      on the given batch.
    """
    pass

  def compute_auxiliary_loss(
      self,
      out,  # pylint: disable=unused-argument
      batch,  # pylint: disable=unused-argument
  ):
    """Compute an auxiliary loss on a single batch.

    Args:
      out: The output of the self-supervised model.
      batch: The output of a VideoDataset dataloader.

    Returns:
      A tensor corresponding to the value of the auxiliary loss function
      evaluated on the given batch.
    """
    return 0.0

  def train_one_iter(self, batch):
    """Single forward + backward pass of the model.

    Args:
      batch: The output of a VideoDataset dataloader.

    Returns:
      A dict of loss values.
    """
    self._model.train()

    self._optimizer.zero_grad()

    # Forward pass to compute embeddings.
    frames = batch["frames"].to(self._device)
    out = self._model(frames)
    # out_dict = self._model(frames)

    # Compute losses.
    loss = self.compute_loss(out.embs, batch)
    aux_loss = self.compute_auxiliary_loss(out, batch)
    # loss = self.compute_loss(out_dict["embs"], batch)
    # aux_loss = self.compute_auxiliary_loss(out_dict, batch)
    total_loss = loss["total_loss"] + aux_loss

    # Backwards pass + optimization step.
    total_loss.backward()
    self._optimizer.step()
    
    def _to_scalar(val):
      if hasattr(val, "item"):
          return val.item()
      elif isinstance(val, (np.ndarray,)):
          return float(np.squeeze(val))
      return val

    return {
    "train/base_loss": _to_scalar(loss["total_loss"]),
    "train/auxiliary_loss": _to_scalar(aux_loss),
    "train/total_loss": _to_scalar(total_loss),
    "train/holdr_loss": _to_scalar(loss["holdr_loss"]),
    "train/contrastive_loss": _to_scalar(loss["contrastive"]),
    "train/distance_subtask_means_loss": _to_scalar(loss["distance_subtask_means_loss"]),
    "train/distance_frames_before_subtask_loss": _to_scalar(loss["distance_frames_before_subtask_loss"]),
  }

  @torch.no_grad()
  def eval_num_iters(
      self,
      valid_loader,
      eval_iters = None,
  ):
    """Compute the loss with the model in `eval()` mode.

    Args:
      valid_loader: The validation data loader.
      eval_iters: The number of time to call `next()` on the data iterator. Set
        to None to evaluate on the whole validation set.

    Returns:
      A dict of validation losses.
    """
    self._model.eval()

    val_base_loss = 0.0
    val_aux_loss = 0.0
    it_ = 0
    for batch_idx, batch in enumerate(valid_loader):
      if eval_iters is not None and batch_idx >= eval_iters:
        break

      frames = batch["frames"].to(self._device)
      out = self._model(frames)
      # out_dict = self._model(frames)
      loss = self.compute_loss(out.embs, batch)
      val_base_loss += loss["total_loss"]
      val_aux_loss += self.compute_auxiliary_loss(out, batch)
      # val_base_loss += self.compute_loss(out_dict["embs"], batch)
      # val_aux_loss += self.compute_auxiliary_loss(out_dict, batch)
      it_ += 1
    val_base_loss /= it_
    val_aux_loss /= it_
    
    # Only convert to scalars for logging/return
    def _to_scalar(val):
        if hasattr(val, "item"):
            return val.item()
        elif isinstance(val, (np.ndarray,)):
            return float(np.squeeze(val))
        return val

    return {
        "valid/base_loss": _to_scalar(val_base_loss),
        "valid/auxiliary_loss": _to_scalar(val_aux_loss),
        "valid/total_loss": _to_scalar(val_base_loss + val_aux_loss),
        "valid/holdr_loss": _to_scalar(loss["holdr_loss"]),
        "valid/contrastive_loss": _to_scalar(loss["contrastive"]),
        "valid/distance_subtask_means_loss": _to_scalar(loss["distance_subtask_means_loss"]),
        "valid/distance_frames_before_subtask_loss": _to_scalar(loss["distance_frames_before_subtask_loss"])
    }
