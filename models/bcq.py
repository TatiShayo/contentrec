import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Any
from data.database import get_db_connection
from data.feedback import get_all_feedback
from data.items import get_all_items

class BCQQNetwork(nn.Module):
    """Discrete two-tower Q-network for session-based recommendations."""

    def __init__(self, state_dim: int = 390, item_emb_dim: int = 384, latent_dim: int = 64):
        super(BCQQNetwork, self).__init__()
        self.state_dim = state_dim
        self.item_emb_dim = item_emb_dim
        self.latent_dim = latent_dim

        # User State Tower
        self.user_tower = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Item Action Tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, state: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) as dot product of latent representations.

        Args:
            state: shape (batch_size, state_dim)
            item_emb: shape (batch_size, item_emb_dim) or (batch_size, num_candidates, item_emb_dim)
        """
        user_latent = self.user_tower(state)  # (batch_size, latent_dim)
        
        if len(item_emb.shape) == 3:
            # item_emb shape: (batch_size, num_candidates, item_emb_dim)
            batch_size, num_cand, _ = item_emb.shape
            item_emb_flat = item_emb.view(-1, self.item_emb_dim)
            item_latent_flat = self.item_tower(item_emb_flat)  # (batch_size * num_cand, latent_dim)
            item_latent = item_latent_flat.view(batch_size, num_cand, self.latent_dim)
            
            # Dot product along latent dimension: (batch_size, num_cand)
            user_latent_unsqueezed = user_latent.unsqueeze(1)  # (batch_size, 1, latent_dim)
            q_values = torch.sum(user_latent_unsqueezed * item_latent, dim=-1)
            return q_values
        else:
            # item_emb shape: (batch_size, item_emb_dim)
            item_latent = self.item_tower(item_emb)  # (batch_size, latent_dim)
            q_values = torch.sum(user_latent * item_latent, dim=-1)
            return q_values


class BCQTrainer:
    """Trainer class for BCQ Q-network extracting transitions from SQLite logs."""

    def __init__(self, model: BCQQNetwork, lr: float = 1e-3, gamma: float = 0.9, batch_size: int = 32):
        self.model = model
        self.target_model = BCQQNetwork(model.state_dim, model.item_emb_dim, model.latent_dim)
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.categories = ["movies", "music", "books", "articles", "news"]

    def _get_category_vector(self, item_id: str, item_map: Dict[str, dict]) -> np.ndarray:
        vec = np.zeros(len(self.categories), dtype=np.float32)
        item = item_map.get(item_id)
        if item:
            cat = (item.get("category") or "").lower()
            if cat in self.categories:
                idx = self.categories.index(cat)
                vec[idx] = 1.0
        return vec

    def prepare_transitions(self, feedback: List[dict], items: List[dict], embedder: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert chronological user sequences into reinforcement learning transitions.

        Returns:
            Tuple of (states, action_embs, rewards, next_states, next_action_embs)
        """
        item_map = {item['item_id']: item for item in items}
        
        # Build pre-computed SBERT embeddings for fast indexing
        item_embeddings = {}
        for item in items:
            if embedder:
                item_embeddings[item['item_id']] = embedder.embed_item(item)
            else:
                item_embeddings[item['item_id']] = np.zeros(384, dtype=np.float32)

        # Group feedback by user and sort chronologically
        from collections import defaultdict
        user_history = defaultdict(list)
        for fb in feedback:
            user_history[fb['user_id']].append(fb)
            
        states, action_embs, rewards, next_states, next_action_embs = [], [], [], [], []

        for user_id, events in user_history.items():
            events.sort(key=lambda x: x.get('timestamp', ''))
            n_events = len(events)
            if n_events < 2:
                continue

            # Accumulate categories for fatigue
            category_counts = defaultdict(int)
            for t in range(n_events - 1):
                # Current state is history up to step t
                history_items = [e['item_id'] for e in events[:t]]
                if history_items:
                    state_emb = np.mean([item_embeddings.get(iid, np.zeros(384)) for iid in history_items], axis=0)
                else:
                    state_emb = np.zeros(384, dtype=np.float32)

                # Fatigue vector
                fatigue_vec = np.zeros(len(self.categories), dtype=np.float32)
                for cat_idx, cat_name in enumerate(self.categories):
                    fatigue_vec[cat_idx] = float(category_counts[cat_name])

                # State step index
                step_idx = np.array([float(t)], dtype=np.float32)
                state_vec = np.concatenate([state_emb, fatigue_vec, step_idx])

                # Action is the item selected at step t
                action_item_id = events[t]['item_id']
                action_emb = item_embeddings.get(action_item_id, np.zeros(384))

                # Update fatigue counts
                item = item_map.get(action_item_id)
                if item and item.get("category"):
                    category_counts[item["category"].lower()] += 1

                # Reward is click (1.0) + scaled dwell time + session continuation (0.5 if there's a next item)
                dwell = events[t].get("dwell_time") or 0.0
                reward = 1.0 + min(dwell * 0.1, 5.0)
                if t < n_events - 2:
                    reward += 0.5

                # Next state is history up to step t+1
                next_history_items = [e['item_id'] for e in events[:t+1]]
                next_state_emb = np.mean([item_embeddings.get(iid, np.zeros(384)) for iid in next_history_items], axis=0)
                
                next_fatigue_vec = np.zeros(len(self.categories), dtype=np.float32)
                for cat_idx, cat_name in enumerate(self.categories):
                    next_fatigue_vec[cat_idx] = float(category_counts[cat_name])
                    
                next_step_idx = np.array([float(t + 1)], dtype=np.float32)
                next_state_vec = np.concatenate([next_state_emb, next_fatigue_vec, next_step_idx])

                # Next action embedding (item chosen at step t+1)
                next_action_item_id = events[t+1]['item_id']
                next_action_emb = item_embeddings.get(next_action_item_id, np.zeros(384))

                states.append(state_vec)
                action_embs.append(action_emb)
                rewards.append(reward)
                next_states.append(next_state_vec)
                next_action_embs.append(next_action_emb)

        if not states:
            # Return empty arrays
            return (np.empty((0, 390)), np.empty((0, 384)), np.empty((0,)), np.empty((0, 390)), np.empty((0, 384)))

        return (
            np.array(states, dtype=np.float32),
            np.array(action_embs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_action_embs, dtype=np.float32)
        )

    def train(self, feedback: List[dict], items: List[dict], embedder: Any, epochs: int = 10) -> List[float]:
        """Train the Q-network offline using Double Q-learning TD updates."""
        states, action_embs, rewards, next_states, next_action_embs = self.prepare_transitions(feedback, items, embedder)
        
        if len(states) == 0:
            logging.info("Insufficient transition data to train BCQ Q-network.")
            return []

        self.model.train()
        losses = []

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, dataset_size, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                
                s_batch = torch.tensor(states[batch_idx], dtype=torch.float32)
                a_emb_batch = torch.tensor(action_embs[batch_idx], dtype=torch.float32)
                r_batch = torch.tensor(rewards[batch_idx], dtype=torch.float32)
                ns_batch = torch.tensor(next_states[batch_idx], dtype=torch.float32)
                na_emb_batch = torch.tensor(next_action_embs[batch_idx], dtype=torch.float32)

                # Predict Q(s, a)
                q_val = self.model(s_batch, a_emb_batch)

                # Compute Target Q-value using target network Q(s', a')
                with torch.no_grad():
                    next_q_val = self.target_model(ns_batch, na_emb_batch)
                    target_q = r_batch + self.gamma * next_q_val

                # TD Loss (MSE)
                loss = nn.functional.mse_loss(q_val, target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Soft update target network: theta_target = tau * theta + (1-tau) * theta_target
            tau = 0.05
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        return losses

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            self.model.load_state_dict(torch.load(filepath, map_location='cpu', weights_only=True))
            self.target_model.load_state_dict(self.model.state_dict())
            self.model.eval()
            return True
        except Exception as e:
            logging.error(f"Error loading BCQ weights: {e}")
            return False
