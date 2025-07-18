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
    text_features
):
    """Embed the stored trajectories and compute mean goal embedding."""
    rews = []
    cosine_similarities = []  # Store cosine similarities for plotting
    continuity_matrices = []  # Store continuity matrix values for plotting
    
    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            print(batch['video_name'])
            frames = batch["frames"].to(device)
            for b in range(frames.shape[0]):
                for t in range(frames.shape[1]):
                    image = frames[b, t]
                    if isinstance(image, np.ndarray):
                        # Convert to torch tensor and permute to (C, H, W)
                        image = torch.from_numpy(image).float().permute(2, 0, 1)  # (C, H, W)
                        image = image / 255.0  # normalize if needed

                    # Add batch and time dimensions: (1, 1, C, H, W)
                    image = image.unsqueeze(0).unsqueeze(0)
                    image = image.to(device)
                    # Forward the pixels through the model and compute the reward.
                    image_features = model.encode_video(image)
                    
                    # Compute raw cosine similarity (like in your other scripts)
                    cosine_sim = cos_sim(image_features, text_features)[0]  # Shape: (3,)
                    cosine_similarities.append(cosine_sim.detach().cpu().numpy())
                    
                    # Compute continuity matrix (transformed cosine similarity)
                    cont_matrix = text_score(image_features, text_features)
                    diag_cont_matrix = cont_matrix[0]
                    continuity_matrices.append(diag_cont_matrix.detach().cpu().numpy())
                    print(f"Cosine similarities: {cosine_sim}")
                    print(f"Diagonal of the continuity matrix: {diag_cont_matrix}")

                    N = text_features.shape[0]
                    eps = 5e-2
                    bias = torch.linspace(eps * (N - 1), 0.0, N, device=diag_cont_matrix.device)
                    diag_cont_matrix += bias
                    target_text_indices = torch.argmax(diag_cont_matrix).item()
                    task_embedding = text_features[target_text_indices]
                    if image_features.dim() == 3:
                        image_features = image_features.squeeze(0)  # (1, D)
                    if image_features.dim() == 1:
                        image_features = image_features.unsqueeze(0)  # (1, D)
                    if task_embedding.dim() == 1:
                        task_embedding = task_embedding.unsqueeze(0)  # (1, D)
                    reward = model.predict_reward([image_features], [task_embedding])
                    if isinstance(reward, torch.Tensor):
                        reward = reward.detach().cpu().item()
                    elif hasattr(reward, "item"):
                        reward = reward.item()
                    rews.append(float(reward))
            break
        break
    return rews, cosine_similarities, continuity_matrices


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

def cos_sim(x1, x2):
        normed_x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
        normed_x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
        return torch.matmul(normed_x1, normed_x2.T)
    
def text_score(image_features, text_features, logit=1.0):
        return (cos_sim(image_features, text_features) + 1) / 2 * logit

def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, downstream_loader = setup()
    model.load_state_dict(torch.load(
            os.path.join(FLAGS.experiment_path, "reds_model.pth"),
            map_location=device,
        ))
    model.to(device).eval()
    rews = []
    print(FLAGS.experiment_path)
    text_phrases = ["The robot moves the red block in the green zone",  "The robot moves the blue block in the green zone" ,  "The robot moves the yellow block in the green zone"]
    text_features = []
    for phrase in text_phrases:
    # Pass as a batch of 1 video, 1 phrase
        text_feature_list = model.encode_text([[phrase]])
        # text_feature_list is a list of 1 tensor of shape (1, D)
        text_feature = text_feature_list[0][0]  # shape: (D,)
        text_features.append(text_feature)
    text_features = torch.stack(text_features, dim=0).to(device)
    
    rews, cosine_similarities, continuity_matrices = embed(model, downstream_loader, device, text_features)
    
    # Ensure all rewards are on CPU and are numpy scalars or floats
    rews = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else float(r) for r in rews]
    # Convert to numpy arrays for plotting
    cosine_similarities = np.array(cosine_similarities)  # Shape: (N_frames, 3)
    continuity_matrices = np.array(continuity_matrices)  # Shape: (N_frames, 3)
    
    # Create subplots for comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rewards over time
    axes[0, 0].plot(rews)
    axes[0, 0].set_title("Reward vs Time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    
    # Plot 2: Cosine similarities for each subtask (like your other scripts)
    for i, phrase in enumerate(text_phrases):
        axes[0, 1].plot(cosine_similarities[:, i], label=f"Subtask {i+1}: {phrase[:20]}...")
    axes[0, 1].set_title("Cosine Similarity vs Time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Cosine Similarity")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Continuity matrix values for each subtask
    for i, phrase in enumerate(text_phrases):
        axes[1, 0].plot(continuity_matrices[:, i], label=f"Subtask {i+1}: {phrase[:20]}...")
    axes[1, 0].set_title("Continuity Matrix Values vs Time")
    axes[1, 0].set_xlabel("Step") 
    axes[1, 0].set_ylabel("Continuity Matrix Value")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    
    plt.tight_layout()

    # Save the comprehensive plot
    save_path = os.path.join("/home/liannello/xirl/experiment_results/Allocentric/training", "ALLO_Reds_Correct.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved comprehensive analysis plot to: {save_path}")
    plt.close()

    # Also save just the cosine similarity plot (similar to your other scripts)
    plt.figure(figsize=(10, 6))
    for i, phrase in enumerate(text_phrases):
        plt.plot(cosine_similarities[:, i], label=f"Subtask {i+1}: {phrase[:30]}...")
    plt.title("Cosine Similarity Between Current Embedding and Subtask Embeddings")
    plt.xlabel("Step")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)

    plt.legend()
    plt.grid(True)

    # Save the cosine similarity plot
    cosine_save_path = os.path.join("/home/liannello/xirl/experiment_results/Allocentric/training", "ALLO_Reds_Cosine_Similarity_Correct.png")
    plt.savefig(cosine_save_path, bbox_inches='tight', dpi=300)
    print(f"Saved cosine similarity plot to: {cosine_save_path}")
    plt.close()

    # Also save just the continuity matrix plot
    plt.figure(figsize=(10, 6))
    for i, phrase in enumerate(text_phrases):
        plt.plot(continuity_matrices[:, i], label=f"Subtask {i+1}: {phrase[:30]}...")
    plt.title("Continuity Matrix Values vs Time")
    plt.xlabel("Step")
    plt.ylabel("Continuity Matrix Value")
    plt.legend()
    plt.grid(True)

    # Save the continuity matrix plot
    continuity_save_path = os.path.join("/home/liannello/xirl/experiment_results/Allocentric/training", "ALLO_Reds_Continuity_Matrix_Correct.png")
    plt.savefig(continuity_save_path, bbox_inches='tight', dpi=300)
    print(f"Saved continuity matrix plot to: {continuity_save_path}")
    plt.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
