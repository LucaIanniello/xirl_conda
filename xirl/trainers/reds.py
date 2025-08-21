from matplotlib import text
import torch
import torch.nn.functional as F
import json
from xirl.trainers.base import Trainer
import os
import numpy as np

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
        loss, epic_loss, supcon_loss = self.compute_loss(video_embs, text_embs, reward, gt_rewards)
        # aux_loss = self.compute_auxiliary_loss(out, batch)
        # loss = self.compute_loss(out_dict["embs"], batch)
        # aux_loss = self.compute_auxiliary_loss(out_dict, batch)
        total_loss = loss

        # Backwards pass + optimization step.
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        
            # Only convert to scalars for logging/return
        def _to_scalar(val):
            if hasattr(val, "item"):
                return val.item()
            elif isinstance(val, (np.ndarray,)):
                return float(np.squeeze(val))
            return val
    
        return {
            "train/base_loss": _to_scalar(loss),
            "train/total_loss": _to_scalar(total_loss),
            "train/epic_loss": _to_scalar(epic_loss),
            "train/supcon_loss": _to_scalar(supcon_loss),   
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
            loss, epic_loss, supcon_loss = self.compute_loss(video_embs, text_embs, reward, gt_rewards)
            val_base_loss += loss
            # val_base_loss += self.compute_loss(out_dict["embs"], batch)
            # val_aux_loss += self.compute_auxiliary_loss(out_dict, batch)
            it_ += 1
        val_base_loss /= it_
        
        # Only convert to scalars for logging/return
        def _to_scalar(val):
            if hasattr(val, "item"):
                return val.item()
            elif isinstance(val, (np.ndarray,)):
                return float(np.squeeze(val))
            return val

        return {
            "valid/base_loss": _to_scalar(val_base_loss),
            "valid/total_loss": _to_scalar(val_base_loss + val_aux_loss),
            "valid/epic_loss": _to_scalar(epic_loss),
            "valid/supcon_loss": _to_scalar(supcon_loss),
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
        return loss, epic_loss, supcon_loss

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
        batch_size = len(pred_rewards)
        losses = []
        for i in range(batch_size):
            pred_i = pred_rewards[i].view(-1)
            gt_i = gt_rewards[i].view(-1)
            
            # Truncate both to the minimal length to avoid shape mismatch
            min_len = min(pred_i.size(0), gt_i.size(0))
            pred_i = pred_i[:min_len]
            gt_i = gt_i[:min_len]
            
            # Canonical set: all other trajectories concatenated
            pred_canon = torch.cat([pred_rewards[j].view(-1)[:min_len] for j in range(batch_size) if j != i])
            gt_canon = torch.cat([gt_rewards[j].view(-1)[:min_len] for j in range(batch_size) if j != i])
            
            # Center by canonical mean
            pred_centered = pred_i - pred_canon.mean()
            gt_centered = gt_i - gt_canon.mean()
            
            loss = self.compute_pearson_distance(pred_centered, gt_centered)
            losses.append(loss)
        return torch.stack(losses).mean()

    
    def supervised_contrastive_loss(self, features, labels, temperature=0.1):
        """
        Args:
            features: (2B, D) tensor (concatenated video and text features)
            labels: (2B,) tensor (labels for each feature)
            temperature: float
        Returns:
            Scalar loss
        """
        device = features.device
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / temperature  # (2B, 2B)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (2B, 2B)

        # Remove self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        # Numerator: exp(sim) for positive pairs
        exp_logits = torch.exp(similarity_matrix) * logits_mask

        # Log prob for each anchor
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Only keep positives
        mask_pos_pairs = mask.sum(dim=1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    def _compute_supcon_loss(self, video_embs, text_embs):
        # video_embs, text_embs: lists of (T, D)
        video_embs_last = torch.stack([v[-1] for v in video_embs])  # (B, D)
        text_embs_last = torch.stack([t[-1] for t in text_embs])    # (B, D)
        features = torch.cat([video_embs_last, text_embs_last], dim=0)  # (2B, D)
        labels = torch.arange(len(video_embs_last), device=video_embs_last.device)
        labels = torch.cat([labels, labels], dim=0)  # (2B,)
        loss = self.supervised_contrastive_loss(features, labels, temperature=self.temperature)
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
    
    
    # def cos_sim(x1, x2):
    #     normed_x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
    #     normed_x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
    #     return torch.matmul(normed_x1, normed_x2.T)
    
    # def text_score(self, image_features, text_features, logit=1.0):
    #     return (self.cos_sim(text_features, image_features) + 1) / 2 * logit
    
    # def reds_reward_step(self, model, image, text_list_encoded):
    #     image_features = model.encode_image(image)
    #     cont_matrix = self.cos_sim(image_features, text_list_encoded)
    #     diag_cont_matrix = torch.diagonal(cont_matrix, dim1=-2, dim2=-1)
        
    #     N = text_list_encoded.shape[0]
    #     eps = 5e-2
    #     bias = torch.linspace(eps * (N - 1), 0.0, N)
    #     diag_cont_matrix += bias
    #     target_text_indices = torch.argmax(diag_cont_matrix).item()
    #     task_embedding = text_list_encoded[target_text_indices]
    #     reward = model.predict_reward(image_features, task_embedding.unsqueeze(0))
    #     return reward