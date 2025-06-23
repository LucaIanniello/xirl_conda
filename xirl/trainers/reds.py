from matplotlib import text
import torch
import torch.nn.functional as F
import json
from xirl.trainers.base import Trainer
import os

class REDSRewardTrainer(Trainer):
    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        reds_cfg = getattr(config.loss, "reds", config.loss)
        
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = getattr(reds_cfg, "supcon_temperature", 0.1)
        self.lambda_supcon = getattr(reds_cfg, "lambda_supcon", 1.0)
        self.lambda_epic = getattr(reds_cfg, "lambda_epic", 1.0)
        self.epic_eps = getattr(reds_cfg, "epic_eps", 5e-2)
        self.lambda_epic_reg = getattr(reds_cfg, "lambda_epic_reg", 1.0)
        
    def train_one_iter(self, batch):
        """Single forward + backward pass of the model.

        Args:
        batch: The output of a VideoDataset dataloader.

        Returns:
        A dict of loss values.
        """
        device = self._device
        self._model.train()

        self._optimizer.zero_grad()

        # Forward pass to compute embeddings.
        frames = batch["frames"].to(self._device)
        # Load ground-truth rewards and texts from files
        gt_rewards, texts, video_names= self._load_gt_and_text(batch, batch["video_name"], device)
        
        reward, video_embs, text_embs = self._model(frames, texts, video_names)
        # out_dict = self._model(frames)

        # Compute losses.
        loss = self.compute_loss(video_embs, text_embs, reward, gt_rewards)
        # aux_loss = self.compute_auxiliary_loss(out, batch)
        # loss = self.compute_loss(out_dict["embs"], batch)
        # aux_loss = self.compute_auxiliary_loss(out_dict, batch)
        total_loss = loss

        # Backwards pass + optimization step.
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
    
        return {
            "train/base_loss": loss,
            "train/total_loss": total_loss,
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
        device = self._device
        
        self._model.eval()

        val_base_loss = 0.0
        val_aux_loss = 0.0
        it_ = 0
        for batch_idx, batch in enumerate(valid_loader):
            if eval_iters is not None and batch_idx >= eval_iters:
                break

            frames = batch["frames"].to(self._device)
            # Load ground-truth rewards and texts from files
            gt_rewards, texts, video_name = self._load_gt_and_text(batch, batch["video_name"], device)
            
            reward, video_embs, text_embs = self._model(frames, texts, video_name)
            # out_dict = self._model(frames)
            val_base_loss += self.compute_loss(video_embs, text_embs, reward, gt_rewards)
            # val_base_loss += self.compute_loss(out_dict["embs"], batch)
            # val_aux_loss += self.compute_auxiliary_loss(out_dict, batch)
            it_ += 1
        val_base_loss /= it_

        return {
            "valid/base_loss": val_base_loss,
            "valid/total_loss": val_base_loss + val_aux_loss,
        }

    def compute_loss(self, video_embs, text_embs, reward, gt_rewards):
        """
        Args:
            batch: dict with keys:
                - 'images': (B, T, C, H, W)
                - 'reward_path': list of file paths to ground-truth reward arrays
                - 'text_path': list of file paths to per-frame text arrays
        Returns:
            loss: scalar tensor
        """
        # Compute losses
        epic_loss = self._compute_epic_loss(reward, gt_rewards)
        supcon_loss = self._compute_supcon_loss(video_embs, text_embs)

        # Combine losses
        loss = self.lambda_epic * epic_loss + self.lambda_supcon * supcon_loss
        return loss

    def extract_score(self, video_features, text_features):
        return self._model.predict_reward(video_features, text_features)
    
    
    def _load_gt_and_text(self, batch, video_names, device):
        gt_rewards = []
        texts = []
        for idx, video_path in enumerate(video_names):
            video_number = os.path.basename(video_path)
            reward_path = os.path.join(video_path, f"{video_number}_sampled_rewards.json")
            text_path = os.path.join(video_path, f"{video_number}_text.json")
            with open(reward_path, "r") as f:
                rewards = torch.tensor(json.load(f), dtype=torch.float32, device=device)
            with open(text_path, "r") as f:
                text = json.load(f)
            # Pad rewards and texts to match the number of frames in batch["frames"]
            n_frames = batch["frames"].shape[1]  # get the number of frames for this video in the batch (should be batch["frames"].shape[1])
            if len(rewards) < n_frames:
                pad_len = n_frames - len(rewards)
                rewards = torch.cat([rewards, torch.zeros(pad_len, device=device)])
                text = text + [""] * pad_len
            gt_rewards.append(rewards)
            texts.append(text)
        return gt_rewards, texts,video_names

    def _compute_epic_loss(self, pred_rewards, gt_rewards):
        # pred_rewards: list of (T, 1)
        # gt_rewards: list of (T,)
        batch_size = len(pred_rewards)
        losses = []
        for i in range(batch_size):
            # Flatten current trajectory
            pred_i = pred_rewards[i].view(-1)
            gt_i = gt_rewards[i].view(-1)
            # Canonical set: all other trajectories in the batch
            pred_canon = torch.cat([pred_rewards[j].view(-1) for j in range(batch_size) if j != i])
            gt_canon = torch.cat([gt_rewards[j].view(-1) for j in range(batch_size) if j != i])
            # Center by canonical mean
            pred_centered = pred_i - pred_canon.mean()
            gt_centered = gt_i - gt_canon.mean()
            # Pearson distance between centered rewards
            loss = self.compute_pearson_distance(pred_centered, gt_centered)
            losses.append(loss)
        # Average over batch
        return torch.stack(losses).mean()

    def _compute_supcon_loss(self, video_embs, text_embs):
        # video_embs, text_embs: lists of (T, D)
        video_embs_last = torch.stack([v[-1] for v in video_embs])  # (B, D)
        text_embs_last = torch.stack([t[-1] for t in text_embs])    # (B, D)
        labels = torch.arange(len(video_embs_last), device=video_embs_last.device)
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