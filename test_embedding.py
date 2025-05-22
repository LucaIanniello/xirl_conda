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

"""Compute and store the mean goal embedding using a trained model."""

import os
import typing

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import utils
from xirl import common
from xirl.models import SelfSupervisedModel
import pdb
import matplotlib.pyplot as plt
from sac.wrappers import HOLDRLearnedVisualReward

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")

ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
    model,
    downstream_loader,
    device,
):
  """Embed the stored trajectories and compute mean goal embedding."""
  for class_name, class_loader in downstream_loader.items():
    logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      print(batch['video_name'])
      out = model.infer(batch["frames"].to(device))
      # out = model.module.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      break
    break   
  return emb


def setup():
  """Load the latest embedder checkpoint and dataloaders."""
  config = utils.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, True)["valid"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    logging.info("Restored model from checkpoint %d.", global_step)
  else:
    logging.info("Skipping checkpoint restore.")
  return model, downstream_loaders

def compute_embedding_distance(emb, goal_emb, subtask_idx, distance_scale):
        dist = np.linalg.norm(emb - goal_emb)
        dist *= distance_scale[subtask_idx]
        return dist
    
def check_subtask_completion(dist, current_reward, subtask, subtask_solved_counter,
                             subtask_threshold, subtask_hold_steps,
                             non_decreasing_reward, num_subtasks,):
        prev_reward = 0.0
        if dist < subtask_threshold:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = min(num_subtasks - 1, subtask + 1)
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
        return prev_reward, subtask, subtask_solved_counter

def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model, downstream_loader = setup()
  model.to(device).eval()
  rews = []
  print(FLAGS.experiment_path)
  if "xirl_embodiment" in FLAGS.experiment_path:
    print("Using distance to goal reward function")
    goal_emb = utils.load_pickle(FLAGS.experiment_path, "goal_emb.pkl")
    distance_scale = utils.load_pickle(FLAGS.experiment_path,
                                            "distance_scale.pkl")
    embs = embed(model, downstream_loader, device)
  
    for emb in embs:
      dist = np.linalg.norm(emb - goal_emb)
      dist = -1.0 * dist * distance_scale
      rews.append(dist)
      
  elif "holdr" in FLAGS.experiment_path:
    print("Using HOLDR reward function")
    subtask_means = utils.load_pickle(FLAGS.experiment_path, "subtask_means.pkl")
    distance_scale = utils.load_pickle(FLAGS.experiment_path, "distance_scale.pkl")
    embs = embed(model, downstream_loader, device)
    subtask = 0
    non_decreasing_reward = False
    prev_reward = 0.0
    subtask_cost = 2.0
    subtask_threshold = 0.2
    subtask_hold_steps = 3
    distance_normalizer = 5
    subtask_solved_counter = 0
    prev_reward = 0.0
    
    for emb in embs:
      current_goal_emb = subtask_means[subtask]
      dist = compute_embedding_distance(emb, current_goal_emb, subtask, distance_scale) 
      shaping = (len(subtask_means) - subtask) * subtask_cost
      if non_decreasing_reward:
        reward = prev_reward + (1.0-dist)
      else:
        reward = - (dist+shaping)/distance_normalizer
      if subtask == 1:
        print("Subtask 1")
      elif subtask == 2:
        print("Subtask 2")
      print(f"Reward: {reward}, Subtask: {subtask}, Distance: {dist}, Shaping: {shaping}")
      rews.append(reward)      
      prev_reward, subtask, subtask_solved_counter = check_subtask_completion(
          dist, reward, subtask, subtask_solved_counter,
          subtask_threshold, subtask_hold_steps,
          non_decreasing_reward, len(subtask_means))
      
  # Save reward plot
  plt.figure()
  plt.plot(rews)
  plt.title("Reward vs Time")
  plt.xlabel("Step")
  plt.ylabel("Reward")
  plt.grid(True)

  # Save the plot instead of showing it
  save_path = os.path.join("/home/lianniello/xirl_thesis/xirl_conda/", "reward_plot_holdr_default.png")
  plt.savefig(save_path, bbox_inches='tight')
  print(f"Saved reward plot to: {save_path}")
  plt.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
