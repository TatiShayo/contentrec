"""Sequential model training loop for SASRec supporting multi-task loss with dwell time."""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class SequentialTrainer:
    """Trains the SASRec sequential recommendation model."""

    def __init__(self, model, lr=1e-3, device='cpu', propensity_estimator=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # allow sample weights
        self.propensity_estimator = propensity_estimator

    def prepare_training_data(
        self, sequences: Dict[str, List], item_to_idx: Dict[str, int], max_seq_len: int = 50, return_dwell: bool = False, return_user_ids: bool = False
    ):
        """Create (input_seq, target) pairs (and optionally dwell_time) using sliding window.

        Supports both standard lists of strings and lists of dicts containing dwell times.
        """
        inputs = []
        targets = []
        dwells = []
        user_ids = []

        for user_id, events in sequences.items():
            if not events:
                continue

            # Handle both list of dicts (with dwell_time) and list of string item IDs
            if isinstance(events[0], dict):
                idx_seq = [item_to_idx.get(e['item_id'], 0) for e in events]
                dwell_seq = [e.get('dwell_time', 0.0) for e in events]
            else:
                idx_seq = [item_to_idx.get(iid, 0) for iid in events]
                dwell_seq = [0.0] * len(events)

            if len(idx_seq) < 2:
                continue

            # Truncate to max_seq_len + 1 (need at least one for target)
            idx_seq = idx_seq[-(max_seq_len + 1):]
            dwell_seq = dwell_seq[-(max_seq_len + 1):]

            # input is all but last, target is all but first, target dwell is all but first
            inp = idx_seq[:-1]
            tgt = idx_seq[1:]
            dwell_tgt = dwell_seq[1:]

            # Pad to max_seq_len
            pad_len = max_seq_len - len(inp)
            inp = [0] * pad_len + inp
            tgt = [0] * pad_len + tgt
            dwell_tgt = [0.0] * pad_len + dwell_tgt

            inputs.append(inp)
            targets.append(tgt)
            dwells.append(dwell_tgt)
            user_ids.append(user_id)

        if not inputs:
            if return_dwell:
                if return_user_ids:
                    return np.array([]), np.array([]), np.array([]), []
                return np.array([]), np.array([]), np.array([])
            if return_user_ids:
                return np.array([]), np.array([]), []
            return np.array([]), np.array([])

        if return_dwell:
            if return_user_ids:
                return (
                    np.array(inputs, dtype=np.int64),
                    np.array(targets, dtype=np.int64),
                    np.array(dwells, dtype=np.float32),
                    user_ids
                )
            return (
                np.array(inputs, dtype=np.int64),
                np.array(targets, dtype=np.int64),
                np.array(dwells, dtype=np.float32)
            )
        if return_user_ids:
            return (
                np.array(inputs, dtype=np.int64),
                np.array(targets, dtype=np.int64),
                user_ids
            )
        return (
            np.array(inputs, dtype=np.int64),
            np.array(targets, dtype=np.int64)
        )

    def train(
        self, sequences: Dict[str, List], item_to_idx: Dict[str, int],
        epochs: int = 20, batch_size: int = 32
    ) -> List[float]:
        """Full training loop. Returns list of epoch losses."""
        inputs, targets, dwells, user_ids = self.prepare_training_data(
            sequences, item_to_idx, max_seq_len=self.model.max_seq_len, return_dwell=True, return_user_ids=True
        )

        if len(inputs) == 0:
            print("No training data for sequential model.")
            return []

        # Build user context mapping and items category mapping for propensity weighting
        user_contexts = {}
        if self.propensity_estimator is not None:
            try:
                from data.database import get_all_impressions
                import json
                impressions = get_all_impressions()
                for imp in impressions:
                    uid = imp["user_id"]
                    cohort = imp["cohort"] or "A"
                    context = {}
                    if imp["context_json"]:
                        try:
                            context = json.loads(imp["context_json"])
                        except:
                            pass
                    device = context.get("device", "desktop")
                    time_of_day = context.get("time_of_day", "afternoon")
                    user_contexts[uid] = (cohort, device, time_of_day)
            except Exception as e:
                import logging
                logging.error(f"Error loading user contexts for online training: {e}")
                
        # Build idx_to_item and items_map for category lookup
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        from data.items import get_all_items
        try:
            items_list = get_all_items()
            items_map = {item["item_id"]: item for item in items_list}
        except:
            items_map = {}

        self.model.train()
        epoch_losses = []

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(inputs))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(inputs), batch_size):
                batch_idx = perm[i:i + batch_size]
                inp_batch = torch.LongTensor(inputs[batch_idx]).to(self.device)
                tgt_batch = torch.LongTensor(targets[batch_idx]).to(self.device)
                dwell_batch = torch.FloatTensor(dwells[batch_idx]).to(self.device)
                batch_uids = [user_ids[idx] for idx in batch_idx]

                self.optimizer.zero_grad()
                logits, mu, sigma = self.model(inp_batch, return_dwell=True)  # (batch, seq_len, num_items)

                # Reshape for cross entropy: (batch*seq_len, num_items) vs (batch*seq_len,)
                logits_flat = logits.view(-1, logits.size(-1))
                tgt_flat = tgt_batch.view(-1)
                
                loss_item_raw = self.criterion(logits_flat, tgt_flat)

                # Apply propensity sample weights if available
                if self.propensity_estimator is not None and items_map:
                    sample_weights = []
                    for b_idx in range(len(batch_uids)):
                        uid = batch_uids[b_idx]
                        cohort, device, time_of_day = user_contexts.get(uid, ("A", "desktop", "afternoon"))
                        
                        tgt_indices = targets[batch_idx[b_idx]]
                        last_tgt_idx = int(tgt_indices[-1])
                        tgt_iid = idx_to_item.get(last_tgt_idx)
                        item = items_map.get(tgt_iid) if tgt_iid else None
                        cat = item.get("category", "") if item else ""
                        
                        weight = self.propensity_estimator.get_ips_weight(cohort, device, time_of_day, cat)
                        sample_weights.extend([weight] * len(tgt_indices))
                        
                    weight_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(self.device)
                    mask_flat = (tgt_flat > 0).float()
                    loss_item = (loss_item_raw * weight_tensor * mask_flat).sum() / (mask_flat.sum() + 1e-10)
                else:
                    mask_flat = (tgt_flat > 0).float()
                    loss_item = (loss_item_raw * mask_flat).sum() / (mask_flat.sum() + 1e-10)

                # Gather predictions for the target items to compute Log-Normal NLL loss
                mu_t = torch.gather(mu, dim=2, index=tgt_batch.unsqueeze(-1)).squeeze(-1)
                sigma_t = torch.gather(sigma, dim=2, index=tgt_batch.unsqueeze(-1)).squeeze(-1)
                
                # Mask out padding targets
                mask = (tgt_batch > 0).float()
                log_y = torch.log(dwell_batch + 1.0)
                
                # Dwell loss step
                loss_dwell_step = log_y + torch.log(sigma_t) + ((log_y - mu_t) ** 2) / (2.0 * (sigma_t ** 2))
                loss_dwell = (loss_dwell_step * mask).sum() / (mask.sum() + 1e-10)

                # Multi-task loss using homoscedastic uncertainty weighting
                loss = (torch.exp(-self.model.log_var_item) * loss_item + 
                        torch.exp(-self.model.log_var_dwell) * loss_dwell + 
                        self.model.log_var_item + self.model.log_var_dwell)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  SASRec Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return epoch_losses

    def save_model(self, path: str):
        """Save model state dict to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_items': self.model.num_items,
            'hidden_dim': self.model.hidden_dim,
            'max_seq_len': self.model.max_seq_len,
            'num_heads': self.model.num_heads,
            'num_blocks': self.model.num_blocks,
        }, path)

    @staticmethod
    def load_model(path: str, num_items: int = None):
        """Load model from disk. Returns (model, checkpoint)."""
        from models.sasrec import SASRec

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        n_items = num_items or checkpoint['num_items']
        model = SASRec(
            num_items=n_items,
            hidden_dim=checkpoint.get('hidden_dim', 64),
            max_seq_len=checkpoint.get('max_seq_len', 50),
            num_heads=checkpoint.get('num_heads', 2),
            num_blocks=checkpoint.get('num_blocks', 2),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint


import queue
import copy
import threading

class OnlineSequentialTrainer:
    """Consumes sequences from a queue and performs real-time online SGD steps."""
    def __init__(self, model_train, model_serve, lr=1e-4, device='cpu', max_buffer_size=1000):
        self.model_train = model_train.to(device)
        self.model_serve = model_serve.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model_train.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.queue = queue.Queue()
        self.replay_buffer = []
        self.max_buffer_size = max_buffer_size
        self._lock = threading.Lock()
        
    def add_sample(self, seq_indices: List[int], next_idx: int, dwell_time: float):
        """Add a single training sample (sequence, next item, dwell time) to the queue."""
        self.queue.put((seq_indices, next_idx, dwell_time))
        
    def process_queue_and_step(self) -> float:
        """Drains the queue, adds to replay buffer, and performs a single SGD step."""
        samples_added = 0
        while not self.queue.empty():
            try:
                seq_indices, next_idx, dwell_time = self.queue.get_nowait()
                if len(seq_indices) == 0 or next_idx == 0:
                    continue
                # input is the sequence, target is next_idx
                # pad to model's max_seq_len
                max_len = self.model_train.max_seq_len
                inp = seq_indices[-max_len:]
                pad_len = max_len - len(inp)
                inp_padded = [0] * pad_len + inp
                
                # target is same shifted by 1, with next_idx at the end
                tgt_padded = inp_padded[1:] + [next_idx]
                dwell_padded = [0.0] * (max_len - 1) + [dwell_time]
                
                self.replay_buffer.append((inp_padded, tgt_padded, dwell_padded))
                samples_added += 1
            except queue.Empty:
                break
                
        # Cap replay buffer
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
            
        if not self.replay_buffer:
            return 0.0
            
        # Draw a batch
        batch_size = min(32, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        batch_inputs = []
        batch_targets = []
        batch_dwells = []
        for idx in indices:
            inp, tgt, dwell = self.replay_buffer[idx]
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_dwells.append(dwell)
            
        self.model_train.train()
        self.optimizer.zero_grad()
        
        inp_tensor = torch.LongTensor(batch_inputs).to(self.device)
        tgt_tensor = torch.LongTensor(batch_targets).to(self.device)
        dwell_tensor = torch.FloatTensor(batch_dwells).to(self.device)
        
        logits, mu, sigma = self.model_train(inp_tensor, return_dwell=True)
        
        # Cross entropy loss
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_flat = tgt_tensor.view(-1)
        loss_item = self.criterion(logits_flat, tgt_flat)
        
        # Dwell loss
        mu_t = torch.gather(mu, dim=2, index=tgt_tensor.unsqueeze(-1)).squeeze(-1)
        sigma_t = torch.gather(sigma, dim=2, index=tgt_tensor.unsqueeze(-1)).squeeze(-1)
        mask = (tgt_tensor > 0).float()
        log_y = torch.log(dwell_tensor + 1.0)
        loss_dwell_step = log_y + torch.log(sigma_t) + ((log_y - mu_t) ** 2) / (2.0 * (sigma_t ** 2))
        loss_dwell = (loss_dwell_step * mask).sum() / (mask.sum() + 1e-10)
        
        # EWC / Multi-task loss using homoscedastic uncertainty log-variances
        loss = (torch.exp(-self.model_train.log_var_item) * loss_item + 
                torch.exp(-self.model_train.log_var_dwell) * loss_dwell + 
                self.model_train.log_var_item + self.model_train.log_var_dwell)
                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_train.parameters(), 5.0)
        self.optimizer.step()
        
        # Atomic weight swap to serve model copy
        with self.model_serve._lock:
            self.model_serve.load_state_dict(copy.deepcopy(self.model_train.state_dict()))
            
        return float(loss.item())


def run_online_sequential_learning(trainer: OnlineSequentialTrainer, stop_event: threading.Event):
    """Background thread loop that executes online SGD steps periodically."""
    logging.info("Online sequential learning thread started.")
    while not stop_event.is_set():
        try:
            # Process and train if samples exist
            loss = trainer.process_queue_and_step()
            if loss > 0.0:
                logging.debug(f"Online incremental SGD loss: {loss:.4f}")
        except Exception as e:
            logging.error(f"Error in online learning step: {e}")
        time.sleep(10.0) # check and update every 10 seconds
    logging.info("Online sequential learning thread stopped.")

