import torch
import torch.nn.functional as F
import json
from xirl.trainers.base import Trainer
import os

class REDSRewardTrainer(Trainer):
    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        reds_cfg = getattr(config.loss, "reds", config.loss)
        self.temperature = getattr(reds_cfg, "supcon_temperature", 0.1)
        self.lambda_supcon = getattr(reds_cfg, "lambda_supcon", 1.0)
        self.lambda_epic = getattr(reds_cfg, "lambda_epic", 1.0)
        self.epic_eps = getattr(reds_cfg, "epic_eps", 5e-2)
        self.lambda_epic_reg = getattr(reds_cfg, "lambda_epic_reg", 1.0)

    def compute_loss(self, batch):
        """
        Args:
            batch: dict with keys:
                - 'images': (B, T, C, H, W)
                - 'reward_path': list of file paths to ground-truth reward arrays
                - 'text_path': list of file paths to per-frame text arrays
        Returns:
            loss: scalar tensor
        """
        device = self._device
        reward_dir = batch["video_name"]
        text_dir = batch["video_name"]
        # Load ground-truth rewards and texts from files
        gt_rewards, texts = self._load_gt_and_text(batch["video_name"], reward_dir, text_dir, device)

        # Forward pass through the model to get predicted rewards and embeddings
        out = self._model(batch["images"].to(device), texts)
        pred_reward = out["pred_reward"].squeeze(-1)  # (B,) or (B, T)
        video_embs = out["video_embs"]                # (B, T, D)
        text_embs = out["text_embs"]                  # (B, T, D)

        # Compute losses
        epic_loss = self._compute_epic_loss(video_embs, text_embs, gt_rewards)
        supcon_loss = self._compute_supcon_loss(video_embs, text_embs)

        # Combine losses
        loss = self.lambda_epic * epic_loss + self.lambda_supcon * supcon_loss
        return loss

    def extract_score(self, video_features, text_features):
        return self._model.predict_reward(video_features, text_features)
    
    
    def _load_gt_and_text(self, video_names, device):
        gt_rewards = []
        texts = []
        for video_path in video_names:
            # Extract the video number (last part of the path)
            video_number = os.path.basename(video_path)
            # Build the paths to the reward and text files
            reward_path = os.path.join(video_path, f"{video_number}_rewards.json")
            text_path = os.path.join(video_path, f"{video_number}_text.json")
            # Load rewards and texts
            with open(reward_path, "r") as f:
                rewards = torch.tensor(json.load(f), dtype=torch.float32, device=device)
            with open(text_path, "r") as f:
                text = torch.tensor(json.load(f), dtype=torch.long, device=device)
            gt_rewards.append(rewards)
            texts.append(text)
        # Stack to get (B, T) or (B, T, ...)
        gt_rewards = torch.stack(gt_rewards)
        texts = torch.stack(texts)
        return gt_rewards, texts

    def _compute_epic_loss(self, video_features, text_features, gt_reward):
        """
        Computes the EPIC loss (Pearson distance) between predicted and ground-truth rewards.
        Args:
            pred_reward: (B, T) or (B,)
            gt_reward: (B, T) or (B,)
        Returns:
            Scalar tensor
        """
        # 1. Predict rewards for current and canonical
        reward = self.extract_score(video_features, text_features).squeeze(-1)      # (B,)
        # 3. Pearson distance
        return self.compute_pearson_distance(reward.flatten(), gt_reward.flatten())

    def _compute_supcon_loss(self, video_embs, text_embs):
        """
        Computes supervised contrastive loss between video and text embeddings.
        Args:
            video_embs: (B, T, D)
            text_embs: (B, T, D)
        Returns:
            Scalar tensor
        """
        # Use last frame for contrastive loss
        video_embs_last = video_embs[:, -1, :]  # (B, D)
        text_embs_last = text_embs[:, -1, :]    # (B, D)
        labels = torch.arange(video_embs_last.size(0), device=video_embs_last.device)
        logits = torch.matmul(F.normalize(video_embs_last, dim=-1), F.normalize(text_embs_last, dim=-1).T) / self.temperature
        loss = F.cross_entropy(logits, labels)
        return loss

    @staticmethod
    def compute_pearson_distance(rewa: torch.Tensor, rewb: torch.Tensor, dist: torch.Tensor = None) -> torch.Tensor:
        if dist is None:
            dist = torch.ones_like(rewa, dtype=rewa.dtype, device=rewa.device) / rewa.numel()
        mean_a = torch.sum(rewa * dist)
        mean_b = torch.sum(rewb * dist)
        rewa_centered = rewa - mean_a
        rewb_centered = rewb - mean_b
        vara = torch.sum((rewa_centered ** 2) * dist)
        varb = torch.sum((rewb_centered ** 2) * dist)
        cov = torch.sum(rewa_centered * rewb_centered * dist)
        corr = cov / (torch.sqrt(vara) * torch.sqrt(varb) + 1e-10)
        corr = torch.clamp(corr, max=1.0)
        return torch.sqrt(0.5 * (1 - corr))