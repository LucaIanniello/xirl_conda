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

def compute_embedding_distance(emb, goal_emb, subtask_idx):
    # dist = np.linalg.norm(emb - goal_emb)
    # # dist *= self._distance_scale[subtask_idx]
    # mod_dist = distance_reward(dist)
    mod_dist = np.dot(emb, goal_emb) / (np.linalg.norm(emb) * np.linalg.norm(goal_emb))
    return 1, mod_dist

def distance_reward(d, alpha=0.00001, beta=0.055, gamma=1e-3):
    return -alpha * d**2 - beta * np.sqrt(d**2 + gamma)

def check_subtask_completion(dist, current_reward, subtask, subtask_solved_counter,
                             subtask_threshold, subtask_hold_steps,
                             non_decreasing_reward, num_subtasks):
    prev_reward = 0.0
    # Logic mirrors the provided _check_subtask_completion
    if subtask == 0:
        if dist > 0.95:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = min(num_subtasks - 1, subtask + 1)
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    elif subtask == 1:
        # Hardcoded threshold for subtask 1, as in your example
        if dist > 0.95:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = min(num_subtasks - 1, subtask + 1)
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    elif subtask == 2:
        # Hardcoded threshold for subtask 2, as in your example
        if dist > 0.95:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = 3
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    # You can add more elifs for further subtasks if needed
    return prev_reward, subtask, subtask_solved_counter

def compute_dense_reward(subtask_means, subtask, emb):
    """Dense reward based on distance to current subtask goal."""
    goal_emb = subtask_means[subtask]
    
    # Raw distance in embedding space
    raw_distance = np.linalg.norm(emb - goal_emb)
    
    # Convert to reward (closer = higher reward)
    # Use exponential decay for smooth gradients
    dense_reward = np.exp(-raw_distance / 2.0)  # Range: [0, 1]
    
    return dense_reward - 0.5  # Center around 0: [-0.5, 0.5]
  
def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model, downstream_loader = setup()
  model.to(device).eval()
  rews = []
  print(FLAGS.experiment_path)
  # if "xirl_embodiment" in FLAGS.experiment_path:
  #   print("Using distance to goal reward function")
  #   goal_emb = utils.load_pickle(FLAGS.experiment_path, "goal_emb.pkl")
  #   distance_scale = utils.load_pickle(FLAGS.experiment_path,
  #                                           "distance_scale.pkl")
  #   embs = embed(model, downstream_loader, device)
  
  #   for emb in embs:
  #     dist = np.linalg.norm(emb - goal_emb)
  #     dist = -1.0 * dist * distance_scale
  #     rews.append(dist)
      
  # # elif "holdr" in FLAGS.experiment_path:
  print("Using HOLDR reward function")
  subtask_means = utils.load_pickle(FLAGS.experiment_path, "subtask_means.pkl")
  distance_scale = utils.load_pickle(FLAGS.experiment_path, "distance_scale.pkl")
  embs = embed(model, downstream_loader, device)
  subtask = 0
  non_decreasing_reward = False
  subtask_cost = 2.0
  subtask_threshold = 5.5 
  subtask_hold_steps = 1
  distance_normalizer = 5
  subtask_solved_counter = 0
  
  cosine_similarity_emb_subtask_1_vec = []
  cosine_similarity_emb_subtask_2_vec = []
  cosine_similarity_emb_subtask_3_vec = []
  previous_reward = -10.0
  
  best_dist_this_subtask = None
  steps_in_current_subtask = 0
  prev_subtask = -1
  transition_phase = False
  transition_steps = 0
  
  i = 0
  for emb in embs: 
    
    subtask_1_mean , subtask_2_mean, subtask_3_mean = subtask_means
          
    cosine_similarity_emb_subtask_1 = np.dot(emb, subtask_1_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_1_mean))
    cosine_similarity_emb_subtask_2 = np.dot(emb, subtask_2_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_2_mean))
    cosine_similarity_emb_subtask_3 = np.dot(emb, subtask_3_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_3_mean))
    
    cosine_similarity_emb_subtask_1_vec.append(cosine_similarity_emb_subtask_1)
    cosine_similarity_emb_subtask_2_vec.append(cosine_similarity_emb_subtask_2)
    cosine_similarity_emb_subtask_3_vec.append(cosine_similarity_emb_subtask_3)
    
    if subtask >= 3:
          reward = subtask * subtask_cost
          print(f"Frame: {i}, Subtask {subtask}, Reward{reward}, Cosine Similarity: {cosine_similarity_emb_subtask_1}, {cosine_similarity_emb_subtask_2}, {cosine_similarity_emb_subtask_3}")
          rews.append(reward)
    else:
          goal_emb = subtask_means[subtask]
          old_dist, dist = compute_embedding_distance(emb, goal_emb, subtask) 

          # if abs(dist) > 1.0:  # Very far (handles transition cases)
          #   base_reward = np.tanh(dist / 2.5)
          # elif abs(dist) > 0.5:  # Far  
          #   base_reward = np.tanh(dist / 1.5)
          # else:  # Close
          #   base_reward = np.tanh(dist / 0.5)
                 
          
          print(f"Frame: {i}, Subtask {subtask},OLDDIST:{old_dist}, Dist:{dist}, Cosine Similarity: {cosine_similarity_emb_subtask_1}, {cosine_similarity_emb_subtask_2}, {cosine_similarity_emb_subtask_3}")
              
          bonus_reward = subtask * subtask_cost
          reward = dist + bonus_reward
          
          # print(f"Reward: {reward}, Subtask: {subtask}, Distance: {dist}")
          rews.append(reward)      
          prev_reward, subtask, subtask_solved_counter = check_subtask_completion(
              dist, reward, subtask, subtask_solved_counter,
              subtask_threshold, subtask_hold_steps,
              non_decreasing_reward, len(subtask_means))
    i += 1
      
  # Save reward plot
  plt.figure()
  plt.plot(rews)
  plt.title("Reward vs Time")
  plt.xlabel("Step")
  plt.ylabel("Reward")
  plt.grid(True)
  
  plt.figure()
  plt.plot(cosine_similarity_emb_subtask_1_vec, label="Subtask 1")
  plt.plot(cosine_similarity_emb_subtask_2_vec, label="Subtask 2")
  plt.plot(cosine_similarity_emb_subtask_3_vec, label="Subtask 3")
  plt.title("Cosine Similarity vs Time")
  plt.xlabel("Step")
  plt.ylabel("Cosine Similarity")
  plt.legend()
  plt.grid(True)
  cosine_save_path = os.path.join("/home/lianniello/xirl_thesis/experiment_results/Egocentric/training_ego", "CosineSimilarity_SubtaskXirl_Wrong.png")
  plt.savefig(cosine_save_path, bbox_inches='tight')
  print(f"Saved cosine similarity plot to: {cosine_save_path}")
  plt.close()

  # Save the plot instead of showing it
  save_path = os.path.join("/home/lianniello/xirl_thesis/experiment_results/Egocentric/training_ego", "ALLO_NEWDIST_TEST.png")
  plt.savefig(save_path, bbox_inches='tight')
  print(f"Saved reward plot to: {save_path}")
  plt.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
