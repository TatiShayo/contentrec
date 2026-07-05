"""BEST-Rec: Universal User Representation via Self-Supervised Pretext Tasks.

Trains a universal sequence encoder over user behavior sequences using three pretext tasks:
1. Masked Item Prediction (MIP)
2. Session Contrastive Loss (SCL)
3. Next Category Prediction (NAP)
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any

class BESTRec(nn.Module):
    def __init__(self, num_items: int, hidden_dim: int = 128, max_seq_len: int = 30, num_categories: int = 5):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_categories = num_categories

        # We add +1 for [MASK] token at index num_items
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pretext heads
        self.mip_head = nn.Linear(hidden_dim, num_items + 2)
        self.nap_head = nn.Linear(hidden_dim, num_categories)
        self.ssl_projection = nn.Linear(hidden_dim, 256)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of:
                - universal user representation (batch_size, 256)
                - sequence item representations (batch_size, seq_len, hidden_dim)
                - category prediction logits (batch_size, num_categories)
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seq) + self.pos_embedding(positions)

        # Transformer encoding
        # No causal mask needed since autoencoders can look bidirectionally (unlike SASRec)
        padding_mask = (seq == 0)
        out = self.transformer(x, src_key_padding_mask=padding_mask)

        # Universal user embedding is the average pooling of the non-padded outputs
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)

        user_emb = F.normalize(self.ssl_projection(pooled), p=2, dim=1)
        nap_logits = self.nap_head(pooled)

        return user_emb, out, nap_logits


class BESTRecTrainer:
    def __init__(self, model: BESTRec, lr: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, sequences: List[List[int]], item_to_cat_idx: Dict[int, int]) -> float:
        """Train one epoch on user interaction sequences using self-supervised pretext losses."""
        if not sequences:
            return 0.0

        self.model.train()
        total_loss = 0.0

        # Construct batch data
        max_len = self.model.max_seq_len
        inputs = []
        next_cats = []

        for seq in sequences:
            if len(seq) < 2:
                continue
            # input is the sequence except last, category target is the category of the last item
            inp = seq[:-1][-max_len:]
            target_item = seq[-1]
            target_cat = item_to_cat_idx.get(target_item, 0)

            # Pad
            pad_len = max_len - len(inp)
            inp = [0] * pad_len + inp

            inputs.append(inp)
            next_cats.append(target_cat)

        if not inputs:
            return 0.0

        X = torch.tensor(inputs, dtype=torch.long).to(self.device)
        y_cat = torch.tensor(next_cats, dtype=torch.long).to(self.device)

        batch_size = 32
        permutation = torch.randperm(len(X))

        for idx in range(0, len(X), batch_size):
            batch_idx = permutation[idx:idx + batch_size]
            seq_batch = X[batch_idx]
            cat_batch = y_cat[batch_idx]

            # 1. Masked Item Prediction (MIP) task: mask 15% of tokens
            # mask token index is num_items
            mask_token = self.model.num_items
            masked_seq = seq_batch.clone()
            
            # Mask generation
            prob_matrix = torch.full(masked_seq.shape, 0.15, device=self.device)
            # Don't mask padding (0)
            prob_matrix.masked_fill_(masked_seq == 0, 0.0)
            mask_indices = torch.bernoulli(prob_matrix).bool()
            
            masked_seq[mask_indices] = mask_token

            # Forward
            self.optimizer.zero_grad()
            
            # View 1: perturbed with masking
            user_emb1, seq_rep1, nap_logits = self.model(masked_seq)
            
            # View 2: original sequence with dropout (SCL contrastive target)
            user_emb2, _, _ = self.model(seq_batch)

            # MIP Loss: Cross-entropy prediction on masked indices only
            mip_logits = self.model.mip_head(seq_rep1)
            mip_loss = F.cross_entropy(
                mip_logits[mask_indices], 
                seq_batch[mask_indices], 
                ignore_index=0
            ) if mask_indices.any() else torch.tensor(0.0, device=self.device)

            # NAP Loss: Cross-entropy next category prediction
            nap_loss = F.cross_entropy(nap_logits, cat_batch)

            # SCL Loss: Cosine similarity contrastive loss (Views 1 & 2)
            # Maximize agreement between different augmentations (masking vs dropout)
            cos_sim = F.cosine_similarity(user_emb1, user_emb2, dim=1)
            scl_loss = (1.0 - cos_sim).mean()

            # Multi-task SSL Loss
            loss = mip_loss + nap_loss + scl_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(seq_batch)

        return total_loss / len(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> bool:
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model.eval()
            return True
        return False
