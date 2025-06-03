import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer
from typing import Dict, List, Union
import pdb

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class HOLDRTrainer(Trainer):
    """Trainer implementing HOLD-R loss.

    The model learns temporal structure by predicting the distance between frames
    in embedding space proportional to their true temporal distance.
    """

    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        self.temperature = config.loss.holdr.temperature if hasattr(config.loss, "holdr") else 1.0

    def compute_loss(self, embs: torch.Tensor, batch: BatchType) -> torch.Tensor:
        """
        Args:
            embs: torch.Tensor of shape (B, T, D), where B is batch size, 
                  T is number of frames per video, D is embedding dimension.
            batch: dict containing at least 'frame_idxs' with shape (B, T)
        """
       
        B, T, D = embs.shape  # embs: [batch_size, num_frames, embedding_dim]
        device = embs.device
        loss = 0.0
        frame_idxs = batch["frame_idxs"].to(device)  # (B, T)
        for i in range(B):
            emb = embs[i]            
            idxs = frame_idxs[i].float()  
                   
            # Compute pairwise embedding distances
            emb_dists = torch.cdist(emb, emb, p=2) / self.temperature
            
            # Compute ground-truth time distances considering the frame indices
            time_dists = torch.cdist(idxs.unsqueeze(1), idxs.unsqueeze(1), p=1)
            
            # Create a mask to consider only upper triangular part of the distance matrix
            # This is to ignore self-distances and lower triangular part
            # We use the upper triangular part because we want to predict distances on the future frames and not
            # considering also the past frames.
            mask = torch.triu(torch.ones_like(time_dists), diagonal=1).bool()

            # Mean squared error between predicted and ground-truth distances
            loss += F.mse_loss(emb_dists[mask], time_dists[mask])
        loss /= B
        return loss
