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

"""Environment wrappers."""

import abc
import collections
import os
import time
import typing

import cv2
import gym
import imageio
import numpy as np
import torch
from xirl.models import SelfSupervisedModel
from xirl.trainers.reds import REDSRewardTrainer

TimeStep = typing.Tuple[np.ndarray, float, bool, dict]
ModelType = SelfSupervisedModel
TensorType = torch.Tensor
DistanceFuncType = typing.Callable[[float], float]
InfoMetric = typing.Mapping[str, typing.Mapping[str, typing.Any]]


class FrameStack(gym.Wrapper):
  """Stack the last k frames of the env into a flat array.

  This is useful for allowing the RL policy to infer temporal information.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, k):
    """Constructor.

    Args:
      env: A gym env.
      k: The number of frames to stack.
    """
    super().__init__(env)

    assert isinstance(k, int), "k must be an integer."

    self._k = k
    self._frames = collections.deque([], maxlen=k)

    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low.min(),
        high=env.observation_space.high.max(),
        shape=((shp[0] * k,) + shp[1:]),
        dtype=env.observation_space.dtype,
    )

  def reset(self):
    obs = self.env.reset()
    for _ in range(self._k):
      self._frames.append(obs)
    return self._get_obs()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._frames.append(obs)
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    assert len(self._frames) == self._k
    return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.

    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      obs, rew, done, info = self.env.step(action)
      total_reward += rew
      if done:
        break
    return obs, total_reward, done, info


class RewardScale(gym.Wrapper):
  """Scale the environment reward."""

  def __init__(self, env, scale):
    """Constructor.

    Args:
      env: A gym env.
      scale: How much to scale the reward by.
    """
    super().__init__(env)

    self._scale = scale

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    reward *= self._scale
    return obs, reward, done, info


class EpisodeMonitor(gym.ActionWrapper):
  """A class that computes episode metrics.

  At minimum, episode return, length and duration are computed. Additional
  metrics that are logged in the environment's info dict can be monitored by
  specifying them via `info_metrics`.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env):
    super().__init__(env)

    self._reset_stats()
    self.total_timesteps: int = 0

  def _reset_stats(self):
    self.reward_sum: float = 0.0
    self.episode_length: int = 0
    self.start_time = time.time()

  def step(self, action):
    obs, rew, done, info = self.env.step(action)

    self.reward_sum += rew
    self.episode_length += 1
    self.total_timesteps += 1
    info["total"] = {"timesteps": self.total_timesteps}

    if done:
      info["episode"] = dict()
      info["episode"]["return"] = self.reward_sum
      info["episode"]["length"] = self.episode_length
      info["episode"]["duration"] = time.time() - self.start_time

    return obs, rew, done, info

  def reset(self):
    self._reset_stats()
    return self.env.reset()


class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      resolution = (256, 256),
      fps = 30,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    self.height, self.width = resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = 0
    self.frames = []

  def step(self, action):
    frame = self.env.render(mode="rgb_array")
    if frame.shape[:2] != (self.height, self.width):
      frame = cv2.resize(
          frame,
          dsize=(self.width, self.height),
          interpolation=cv2.INTER_CUBIC,
      )
    self.frames.append(frame)
    observation, reward, done, info = self.env.step(action)
    if done:
      filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
      imageio.mimsave(filename, self.frames, fps=self.fps)
      self.frames = []
      self.current_episode += 1
    return observation, reward, done, info


# ========================================= #
# Learned reward wrappers.
# ========================================= #

# Note: While the below classes provide a nice wrapper API, they are not
# efficient for training RL policies as rewards are computed individually at
# every `env.step()` and so cannot take advantage of batching on the GPU.
# For actually training policies, it is better to use the learned replay buffer
# implementations in `sac.replay_buffer.py`. These store transitions in a
# staging buffer which is forwarded as a batch through the GPU.


class LearnedVisualReward(abc.ABC, gym.Wrapper):
  """Base wrapper class that replaces the env reward with a learned one.

  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      env,
      model,
      device,
      res_hw = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      device: Compute device.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._res_hw = res_hw

  def _to_tensor(self, x):
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    # TODO(kevin): Make this more generic for other preprocessing.
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self):
    """Render the pixels at the desired resolution."""
    # TODO(kevin): Make sure this works for mujoco envs.
    pixels = self.env.render(mode="rgb_array")
    if self._res_hw is not None:
      h, w = self._res_hw
      pixels = cv2.resize(pixels, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return pixels

  @abc.abstractmethod
  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    # We'll keep the original env reward in the info dict in case the user would
    # like to use it in conjunction with the learned reward.
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    return obs, learned_reward, done, info


class DistanceToGoalLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb,
      distance_scale = 1.0,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      goal_emb: The goal embedding.
      distance_scale: Scales the distance from the current state embedding to
        that of the goal state. Set to `1.0` by default.
      **base_kwargs: Base keyword arguments.
    """
    super().__init__(**base_kwargs)

    self._goal_emb = np.atleast_2d(goal_emb)
    self._distance_scale = distance_scale

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    # print("Computing reward from image dist.")
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    # emb = self._model.module.infer(image_tensor).numpy().embs
    dist = -1.0 * np.linalg.norm(emb - self._goal_emb)
    dist *= self._distance_scale
    return dist


class GoalClassifierLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with the output of a goal classifier."""

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    print("Computing reward from image.")
    image_tensor = self._to_tensor(image)
    prob = torch.sigmoid(self._model.infer(image_tensor).embs)
    # prob = torch.sigmoid(self._model.module.infer(image_tensor).embs)
    return prob.item()

class HOLDRLearnedVisualReward(LearnedVisualReward):
    def __init__(
        self,
        subtask_means,
        distance_scale,
        subtask_threshold=5.0,
        subtask_cost=3.0,
        subtask_hold_steps=1,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self._subtask_means = np.atleast_2d(subtask_means)  
        self._distance_scale = distance_scale               
        self._num_subtasks = len(subtask_means) + 1

        # Subtask tracking
        self._subtask = 0
        self._subtask_threshold = subtask_threshold
        self._subtask_cost = subtask_cost
        self._subtask_hold_steps = subtask_hold_steps
        self._subtask_solved_counter = 0
        self._non_decreasing_reward = False
        self._prev_reward= 0.0
        
        self._distance_normalizer = 5
        
                
    def reset_state(self):
        # print("Resetting HOLDR wrapper.")
        self._subtask = 0
        self._subtask_solved_counter = 0
        self._prev_reward = 0.0
       
    def _compute_embedding_distance(self, emb, goal_emb, subtask_idx):
        dist = np.linalg.norm(emb - goal_emb)
        # dist *= self._distance_scale[subtask_idx]
        dist = self._distance_reward(dist)
        return dist
    
    def _distance_reward(self, d, alpha=0.001, beta=0.01, gamma=1e-3):
      return -alpha * d**2 - beta * np.sqrt(d**2 + gamma)
    
    def _check_subtask_completion(self, dist, current_reward):
      if self._subtask == 0:
        if dist > -0.1:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 1:
        if dist > -0.15:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 2:
        if dist > -0.25:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
      
            
    def _get_reward_from_image(self, image):
       
        
        image_tensor = self._to_tensor(image)
        emb = self._model.infer(image_tensor).numpy().embs  # Shape: (emb_dim,)
        # emb = self._model.module.infer(image_tensor).numpy().embs
        
        # dists = [np.linalg.norm(emb - mean) for mean in self._subtask_means]
        # self._subtask = np.argmin(dists)
        # subtasks.append(self._subtask)
        if self._subtask >= self._num_subtasks - 1:
          reward = self._subtask_cost * self._subtask
        else:
          goal_emb = self._subtask_means[self._subtask]
        
          # Distance-based reward
          dist = self._compute_embedding_distance(emb, goal_emb, self._subtask)
          # print(f"Subtask {self._subtask}, Distance: {dist}")

          # shaping = (self._num_subtasks - self._subtask) * self._subtask_cost
          # goal_dist = self._compute_embedding_distance(goal_emb, goal_emb, self._subtask)
          
          # print(f"Subtask {self._subtask}, Distance: {dist}, Goal Distance: {goal_dist}")
          # if self._non_decreasing_reward:
          # reward = self._prev_reward + (1.0 - dist)
          # else:
          # reward = - max(0.0, dist - goal_dist) / self._distance_normalizer
          # reward = - (dist + shaping) / self._distance_normalizer
          
          step_reward = dist
          bonus_reward = self._subtask * self._subtask_cost
          reward = step_reward + bonus_reward
          

            
          # print(f"Subtask {self._subtask}, Reward: {reward}, Distance: {dist}, Previous Reward: {self._previous_reward}")
          
          #Normalization
          # reward = (reward / 6.0) - 1.0
              
          # if self._subtask == 1:
          #         print("Subtask 1 completed, reward:", reward)
          # elif self._subtask == 2:
          #         print("Subtask 2 completed, reward:", reward)
              
          # Check if the subtask is completed
          self._check_subtask_completion(dist, reward)
            
        return reward
        
class REDSLearnedVisualReward(LearnedVisualReward):
    """Replace the environment reward with the output of a REDS model."""

    def __init__(
        self,
        **base_kwargs,
    ):
        """Constructor."""
        super().__init__(**base_kwargs)
        self.text_phrases = ["The robot moves the red block in the goal zone",  "The robot moves the blue block in the goal zone" ,  "The robot moves the yellow block in the goal zone"]
        self.text_features = []
        for phrase in self.text_phrases:
          # Pass as a batch of 1 video, 1 phrase
          text_feature_list = self._model.encode_text([[phrase]])
          # text_feature_list is a list of 1 tensor of shape (1, D)
          text_feature = text_feature_list[0][0]  # shape: (D,)
          self.text_features.append(text_feature)
        self.text_features = torch.stack(self.text_features, dim=0).to(self._device)
        
    def cos_sim(self, x1, x2):
        normed_x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
        normed_x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
        return torch.matmul(normed_x1, normed_x2.T)
    
    def text_score(self, image_features, text_features, logit=1.0):
        return (self.cos_sim(image_features, text_features) + 1) / 2 * logit
      
    def _get_reward_from_image(self, image):
        if isinstance(image, np.ndarray):
            # Convert to torch tensor and permute to (C, H, W)
            image = torch.from_numpy(image).float().permute(2, 0, 1)  # (C, H, W)
            image = image / 255.0  # normalize if needed

        # Add batch and time dimensions: (1, 1, C, H, W)
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.to(self._device)
        # print("Computing reward from image.")
        """Forward the pixels through the model and compute the reward."""
        image_features = self._model.encode_video(image)
        cont_matrix = self.text_score(image_features, self.text_features)
        diag_cont_matrix = cont_matrix[0]

        N = self.text_features.shape[0]
        eps = 5e-2
        bias = torch.linspace(eps * (N - 1), 0.0, N, device=diag_cont_matrix.device)
        diag_cont_matrix += bias
        target_text_indices = torch.argmax(diag_cont_matrix).item()
        task_embedding = self.text_features[target_text_indices]
        if image_features.dim() == 3:
            image_features = image_features.squeeze(0)  # (1, D)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)  # (1, D)
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)  # (1, D)
        reward = self._model.predict_reward([image_features], [task_embedding])
        return reward[0].item() if reward[0].numel() == 1 else reward[0]
        


  

