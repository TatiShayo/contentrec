import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
from data.database import get_db_connection

class MLPEncoder(nn.Module):
    def __init__(self, input_dim=73, feature_dim=16):
        super(MLPEncoder, self).__init__()
        # Fix the seed for reproducible projection
        torch.manual_seed(42)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim)
        )
        # Non-trainable random projection or small feature extractor
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.fc(x)

class NeuralLinearBandit:
    def __init__(self, context_dim=73, feature_dim=16, alpha=0.5, discount=0.99):
        self.context_dim = context_dim
        self.feature_dim = feature_dim
        self.alpha = alpha  # Thompson sampling exploration parameter
        self.discount = discount
        self.encoder = MLPEncoder(input_dim=context_dim, feature_dim=feature_dim)
        
        # 6 weights: w_rel, w_fresh, w_fatigue, w_context, w_rl, w_ssl
        self.arms = {
            0: (1.0, 0.2, 0.3, 0.4, 0.1, 0.15), # Default
            1: (1.0, 0.5, 0.1, 0.2, 0.05, 0.1), # Freshness-heavy
            2: (0.5, 0.1, 0.8, 0.2, 0.15, 0.2), # Anti-fatigue-heavy
            3: (0.8, 0.1, 0.2, 0.8, 0.05, 0.1), # Context-heavy
            4: (1.2, 0.1, 0.1, 0.1, 0.2, 0.25)  # Relevance-heavy
        }
        self.num_actions = len(self.arms)
        
        # Initialize Bayesian regression parameters for each arm
        # B_a = covariance inverse (precision matrix), f_a = cumulative reward vector
        self.B = {a: np.eye(feature_dim) for a in range(self.num_actions)}
        self.f = {a: np.zeros(feature_dim) for a in range(self.num_actions)}
        self.mu = {a: np.zeros(feature_dim) for a in range(self.num_actions)}
        
        self.load_state()

    def get_phi(self, context_vector: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0)
        phi = self.encoder(tensor).squeeze(0).numpy()
        return phi

    def select_action(self, context_vector: np.ndarray) -> Tuple[int, Tuple[float, float, float, float]]:
        phi = self.get_phi(context_vector)
        sampled_scores = []
        
        for a in range(self.num_actions):
            # Sigma_a = B_a^-1
            Sigma = np.linalg.inv(self.B[a])
            # Thompson sampling: sample beta_a from N(mu_a, alpha^2 * Sigma_a)
            beta_sampled = np.random.multivariate_normal(self.mu[a], (self.alpha ** 2) * Sigma)
            score = np.dot(phi, beta_sampled)
            sampled_scores.append(score)
            
        best_arm = int(np.argmax(sampled_scores))
        return best_arm, self.arms[best_arm]

    def update(self, arm_id: int, context_vector: np.ndarray, reward: float):
        phi = self.get_phi(context_vector)
        
        # Apply discount to prevent posterior collapse/forget older sessions
        self.B[arm_id] = self.discount * self.B[arm_id] + (1 - self.discount) * np.eye(self.feature_dim)
        self.f[arm_id] = self.discount * self.f[arm_id]
        
        # Bayesian update:
        # B_a <- B_a + phi * phi^T
        self.B[arm_id] += np.outer(phi, phi)
        # f_a <- f_a + r * phi
        self.f[arm_id] += reward * phi
        # mu_a <- B_a^-1 * f_a
        self.mu[arm_id] = np.linalg.inv(self.B[arm_id]).dot(self.f[arm_id])
        
        self.save_state()

    def save_state(self):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for a in range(self.num_actions):
                    state = {
                        'B': self.B[a].tolist(),
                        'f': self.f[a].tolist(),
                        'mu': self.mu[a].tolist()
                    }
                    cursor.execute(
                        "INSERT OR REPLACE INTO bandit_states (arm_id, state_json) VALUES (?, ?)",
                        (str(a), json.dumps(state))
                    )
                conn.commit()
        except Exception as e:
            logging.error(f"Error saving bandit state to DB: {e}")

    def load_state(self):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT arm_id, state_json FROM bandit_states")
                rows = cursor.fetchall()
                for row in rows:
                    arm_id = int(row['arm_id'])
                    state = json.loads(row['state_json'])
                    self.B[arm_id] = np.array(state['B'])
                    self.f[arm_id] = np.array(state['f'])
                    self.mu[arm_id] = np.array(state['mu'])
        except Exception as e:
            logging.warning(f"Could not load bandit state from DB (may be empty): {e}")
