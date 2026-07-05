"""SASRec: Self-Attentive Sequential Recommendation model with Dwell-Time prediction.

A PyTorch implementation of the SASRec architecture for next-item prediction,
extended with a dwell-time regression head (Log-Normal distribution) and
homoscedastic uncertainty weighting for multi-task training.
"""

import math
import threading
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Embeds item sequences with learned positional encodings, then processes
    them through a stack of Transformer encoder blocks with causal attention
    masks to predict the next item and its expected dwell-time.

    Args:
        num_items: Total number of items (including padding at index 0).
        hidden_dim: Dimensionality of item/positional embeddings.
        max_seq_len: Maximum input sequence length.
        num_heads: Number of attention heads per Transformer block.
        num_blocks: Number of stacked Transformer encoder blocks.
        dropout: Dropout rate for embeddings and attention.
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self._lock = threading.Lock()

        # Embedding layers – padding_idx=0 so pad tokens get zero vectors
        self.item_embedding = nn.Embedding(
            num_items, hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks
        )

        # Output projection to item logits (next item head)
        self.output_layer = nn.Linear(hidden_dim, num_items)

        # Dwell time prediction heads (regression: Log-Normal mu and sigma)
        self.dwell_mu = nn.Linear(hidden_dim, num_items)
        self.dwell_sigma_raw = nn.Linear(hidden_dim, num_items)

        # Homoscedastic uncertainty loss parameters
        self.log_var_item = nn.Parameter(torch.zeros(1))
        self.log_var_dwell = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform initialization for embeddings and output layers."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # skip pad
        nn.init.xavier_uniform_(self.position_embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        nn.init.xavier_uniform_(self.dwell_mu.weight)
        nn.init.zeros_(self.dwell_mu.bias)
        nn.init.xavier_uniform_(self.dwell_sigma_raw.weight)
        nn.init.zeros_(self.dwell_sigma_raw.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate an upper-triangular causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(self, seq: torch.Tensor, return_dwell: bool = False):
        """Forward pass: item sequence -> next-item logits.

        Args:
            seq: Integer tensor of shape (batch_size, seq_len) containing
                 item indices. Padding uses index 0.
            return_dwell: If True, also return predicted dwell time mu and sigma parameters.

        Returns:
            If return_dwell is False:
              - Logits tensor of shape (batch_size, seq_len, num_items)
            If return_dwell is True:
              - Tuple of (logits, mu, sigma)
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        # Position indices: [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Combine item + positional embeddings
        x = self.item_embedding(seq) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        # Causal mask so position i can only attend to positions <= i
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Padding mask: True where seq == 0 (padding token)
        padding_mask = (seq == 0)

        # Transformer forward
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        # Project to item logits
        logits = self.output_layer(x)
        
        if not return_dwell:
            return logits

        mu = self.dwell_mu(x)
        sigma = F.softplus(self.dwell_sigma_raw(x)) + 1e-4

        return logits, mu, sigma

    @torch.no_grad()
    def predict_next(self, seq: List[int], n: int = 10, beta: float = 0.1) -> List[int]:
        """Predict the top-N most likely next items for a given sequence, blending dwell-time expectation.

        Thread-safe: acquires an internal lock before running inference.

        Args:
            seq: List of item indices representing the user's history.
            n: Number of top items to return.
            beta: Blending weight for expected dwell time. Set to 0.0 for pure next-item logits.

        Returns:
            List of top-N item indices sorted by descending probability/blended score.
        """
        with self._lock:
            self.eval()

            if not seq:
                return []

            # Truncate to max_seq_len
            seq = seq[-self.max_seq_len:]

            # Pad from the left if shorter than max_seq_len
            pad_len = self.max_seq_len - len(seq)
            padded = [0] * pad_len + seq

            input_tensor = torch.tensor([padded], dtype=torch.long)
            logits, mu, sigma = self.forward(input_tensor, return_dwell=True)

            # Take predictions from the last non-padding position
            last_logits = logits[0, -1, :].clone()  # (num_items,)
            last_mu = mu[0, -1, :]                  # (num_items,)
            last_sigma = sigma[0, -1, :]            # (num_items,)

            # Zero out padding index so it's never recommended
            last_logits[0] = float("-inf")

            # Also zero out items already in the sequence to avoid repeats
            for idx in seq:
                if idx > 0:
                    last_logits[idx] = float("-inf")

            # Log probability
            log_probs = F.log_softmax(last_logits, dim=0)

            if beta > 0.0:
                # E[dwell] = exp(mu + sigma^2 / 2) - 1
                expected_dwell = torch.exp(last_mu + (last_sigma ** 2) / 2.0) - 1.0
                scores = log_probs + beta * expected_dwell
            else:
                scores = log_probs

            top_k = torch.topk(scores, self.num_items)

            # Extract indices that are not padding and not in the history
            results = []
            history_set = set(seq)
            for idx in top_k.indices.tolist():
                if idx != 0 and idx not in history_set:
                    results.append(idx)
                    if len(results) == n:
                        break

            return results
