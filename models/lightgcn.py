import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import logging

class LightGCN(nn.Module):
    """Natively implemented LightGCN model in PyTorch with side-information projection."""
    def __init__(self, num_users, num_items, emb_dim=64, num_layers=2, reg=1e-4):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.reg = reg
        
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        
        # Initialize weights
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
    def initialize_items_with_features(self, item_features: np.ndarray):
        """Warm-start item embeddings with projected SBERT text embeddings."""
        feature_dim = item_features.shape[1]
        proj = nn.Linear(feature_dim, self.emb_dim)
        with torch.no_grad():
            feat_tensor = torch.tensor(item_features, dtype=torch.float32)
            projected = proj(feat_tensor)
            self.item_emb.weight.copy_(projected)

    def get_adj_matrix(self, interactions_list):
        """Construct normalized adjacency matrix dynamically."""
        R = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        for u, i in interactions_list:
            R[u, i] = 1.0
            
        # Adjacency matrix A = [[0, R], [R^T, 0]]
        adj = np.zeros((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj[:self.num_users, self.num_users:] = R
        adj[self.num_users:, :self.num_users] = R.T
        
        # Degrees
        rowsum = np.sum(adj, axis=1)
        d_inv = np.power(rowsum, -0.5, where=rowsum > 0)
        d_inv[np.isinf(d_inv) | np.isnan(d_inv)] = 0.0
        D_inv = np.diag(d_inv)
        
        # L = D^-0.5 A D^-0.5
        L = D_inv.dot(adj).dot(D_inv)
        return torch.tensor(L, dtype=torch.float32)

    def forward(self, L):
        """Propagate embeddings across layers."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        
        for _ in range(self.num_layers):
            all_emb = torch.matmul(L, all_emb)
            embs.append(all_emb)
            
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        users, items = torch.split(final_emb, [self.num_users, self.num_items], dim=0)
        return users, items

    def bpr_loss(self, users_emb, items_emb, u, i, j):
        """BPR pairwise ranking loss with regularization."""
        u_emb = users_emb[u]
        pos_emb = items_emb[i]
        neg_emb = items_emb[j]
        
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        loss = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))
        
        # Regularization
        reg_loss = (self.user_emb(u).norm(2).pow(2) + 
                    self.item_emb(i).norm(2).pow(2) + 
                    self.item_emb(j).norm(2).pow(2)) / float(len(u))
        
        return loss + self.reg * reg_loss


class LightGCNTrainer:
    """Helper class to orchestrate dataset preparation and training for LightGCN."""
    def __init__(self, emb_dim=64, lr=1e-3, epochs=20, batch_size=64):
        self.emb_dim = emb_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}

    def train_model(self, feedback, items, text_embedder=None):
        """Orchestrate training of LightGCN."""
        if not feedback or not items:
            return None, None
            
        # Build index mapping
        user_ids = sorted(list(set(f['user_id'] for f in feedback)))
        item_ids = sorted([i['item_id'] for i in items])
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(item_ids)}
        
        num_users = len(user_ids)
        num_items = len(item_ids)
        
        # Create model
        self.model = LightGCN(num_users, num_items, emb_dim=self.emb_dim)
        
        # Initialize with side information features if SBERT is provided
        if text_embedder is not None:
            features = []
            for iid in item_ids:
                # Find item details
                matching_item = next((item for item in items if item['item_id'] == iid), None)
                if matching_item:
                    emb = text_embedder.embed_item(matching_item)
                    features.append(emb)
                else:
                    features.append(np.zeros(text_embedder.dimension, dtype=np.float32))
            self.model.initialize_items_with_features(np.array(features))
            
        # Prepare interaction list
        interactions = []
        for f in feedback:
            if f['user_id'] in self.user_to_idx and f['item_id'] in self.item_to_idx:
                interactions.append((self.user_to_idx[f['user_id']], self.item_to_idx[f['item_id']]))
                
        if not interactions:
            return None, None
            
        L = self.model.get_adj_matrix(interactions)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Positive interaction lookup for negative sampling
        pos_set = set(interactions)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            # Sample triplets (user, pos_item, neg_item)
            triplets = []
            for u_idx, i_idx in interactions:
                # Sample negative item
                for _ in range(5): # 5 negative samples per positive
                    j_idx = np.random.randint(0, num_items)
                    if (u_idx, j_idx) not in pos_set:
                        triplets.append((u_idx, i_idx, j_idx))
                        break
            
            if not triplets:
                continue
                
            np.random.shuffle(triplets)
            
            # Batch updates
            for b in range(0, len(triplets), self.batch_size):
                batch = triplets[b:b+self.batch_size]
                u_batch = torch.tensor([x[0] for x in batch], dtype=torch.long)
                i_batch = torch.tensor([x[1] for x in batch], dtype=torch.long)
                j_batch = torch.tensor([x[2] for x in batch], dtype=torch.long)
                
                optimizer.zero_grad()
                users_emb, items_emb = self.model(L)
                loss = self.model.bpr_loss(users_emb, items_emb, u_batch, i_batch, j_batch)
                loss.backward()
                optimizer.step()
                
        # Generate final representations
        self.model.eval()
        with torch.no_grad():
            users_emb, items_emb = self.model(L)
            
        return users_emb.numpy(), items_emb.numpy()

    def save(self, filepath, users_emb=None, items_emb=None):
        """Save the trainer state mapping and model weights, including final embeddings."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict() if self.model else None,
                'user_to_idx': self.user_to_idx,
                'idx_to_user': self.idx_to_user,
                'item_to_idx': self.item_to_idx,
                'idx_to_item': self.idx_to_item,
                'emb_dim': self.emb_dim,
                'users_emb': users_emb,
                'items_emb': items_emb
            }, f)

    def load(self, filepath):
        """Load the trainer state mapping and model weights, including final embeddings."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.user_to_idx = state['user_to_idx']
            self.idx_to_user = state['idx_to_user']
            self.item_to_idx = state['item_to_idx']
            self.idx_to_item = state['idx_to_item']
            self.emb_dim = state['emb_dim']
            self.users_emb = state.get('users_emb', None)
            self.items_emb = state.get('items_emb', None)
            
            num_users = len(self.user_to_idx)
            num_items = len(self.item_to_idx)
            if num_users > 0 and num_items > 0:
                self.model = LightGCN(num_users, num_items, emb_dim=self.emb_dim)
                if state['model_state'] is not None:
                    self.model.load_state_dict(state['model_state'])
                return True
        return False
