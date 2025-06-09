import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer
from typing import Dict, List, Union
import pdb
import json
import os
from collections import defaultdict

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class HOLDRTrainer(Trainer):
    """Trainer implementing HOLD-R loss.

    The model learns temporal structure by predicting the distance between frames
    in embedding space proportional to their true temporal distance.
    """

    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        self.temperature = config.loss.holdr.temperature if hasattr(config.loss, "holdr") else 1.0
        self.distance_subtask_means_weight = config.loss.holdr.distance_subtask_means_weight if hasattr(config.loss, "holdr") else 0.0
        self.distance_frames_before_subtask_weight = config.loss.holdr.distance_frames_before_subtask_weight if hasattr(config.loss, "holdr") else 0.0
        with open(config.loss.holdr.subtask_json_path, "r") as f:
            raw = json.load(f)

        self.subtask_map = {}
        for vid_id, frame_paths in raw.items():
            for subtask_id, frame_path in enumerate(frame_paths):
                # Extract frame idx (e.g., '32.png' -> 32)
                frame_idx = int(os.path.basename(frame_path).split(".")[0])
                video_name = "/".join(frame_path.split("/")[-3:-1])  # e.g., "gripper/0"
                self.subtask_map[(video_name, frame_idx)] = subtask_id

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
        holdr_loss = 0.0
        frame_idxs = batch["frame_idxs"].to(device)  # (B, T)
        
        distance_subtask_means_loss = 0.0
        distance_frames_before_subtask_loss = 0.0
        subtask_embeddings = defaultdict(list)
        
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
            holdr_loss += F.mse_loss(emb_dists[mask], time_dists[mask])
            
            #SEMANTIC LOSS
            vid_path = batch["video_name"][i]
            video_name = "/".join(vid_path.split("/")[-2:])  # e.g., "gripper/0"
            
            for j,t in enumerate(idxs):
                key = (video_name, int(t.item()))
                if key in self.subtask_map:
                    subtask_id = self.subtask_map[key]
                    subtask_embeddings[subtask_id].append(emb[j])
               
        subtask_means = {}     
        for subtask_id, emb_list in subtask_embeddings.items():
            if len(emb_list) < 2:
                continue
            embs_tensor = torch.stack(emb_list, dim=0)
            mean_emb = embs_tensor.mean(dim=0)
            subtask_means[subtask_id] = mean_emb
            for e in emb_list:
                target = torch.tensor([1.0], device=e.device)  # label: similar
                distance_subtask_means_loss += F.cosine_embedding_loss(
                    e.unsqueeze(0),
                    mean_emb.unsqueeze(0),
                    target
                )
                
        for i in range(B):
            emb = embs[i]         
            idxs = frame_idxs[i]     
            vid_path = batch["video_name"][i]
            video_name = "/".join(vid_path.split("/")[-2:])

            for j, t in enumerate(idxs):
                t_int = int(t.item())
                key = (video_name, t_int)

                if key in self.subtask_map:
                    subtask_id = self.subtask_map[key]
                    for k in range(j): 
                        prev_emb = emb[k]
                        target = torch.tensor([1.0], device=prev_emb.device)
                        distance_frames_before_subtask_loss += F.cosine_embedding_loss(
                            prev_emb.unsqueeze(0),
                            subtask_means[subtask_id].unsqueeze(0),
                            target
                        )         
        
        holdr_loss /= B
        distance_subtask_means_loss /= max(1, len(subtask_embeddings))
        distance_frames_before_subtask_loss /= max(1, len(subtask_embeddings))
                
        loss = holdr_loss + self.distance_subtask_means_weight * distance_subtask_means_loss + self.distance_frames_before_subtask_weight * distance_frames_before_subtask_loss
        return loss
