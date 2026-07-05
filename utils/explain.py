import numpy as np
import torch
from typing import List, Dict, Any, Tuple

class LinearShapleyExplainer:
    """Computes exact Shapley attribution values for linear multi-objective recommendation scores."""

    @staticmethod
    def explain_recommendation(
        item_id: str,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        weights: Dict[str, float],
        context_device: str = None,
        context_location: str = None,
        context_time: str = None
    ) -> Tuple[float, Dict[str, float], str]:
        """Compute exact Shapley attribution values compared to average candidate pool.

        Score = w_rel * x_rel + w_fresh * x_fresh - w_fatigue * x_fatigue + w_context * x_context + w_ssl * x_ssl
        """
        w_rel = weights.get('w_relevance', 1.0)
        w_fresh = weights.get('w_freshness', 0.2)
        w_fatigue = weights.get('w_fatigue', 0.3)
        w_context = weights.get('w_context', 0.4)
        w_ssl = weights.get('w_ssl', 0.15)

        # Compute differences from baseline
        diff_rel = metrics.get('relevance', 0.0) - baseline_metrics.get('relevance', 0.0)
        diff_fresh = metrics.get('freshness', 0.0) - baseline_metrics.get('freshness', 0.0)
        diff_fatigue = metrics.get('fatigue', 0.0) - baseline_metrics.get('fatigue', 0.0)
        diff_context = metrics.get('context', 0.0) - baseline_metrics.get('context', 0.0)
        diff_ssl = metrics.get('ssl', 0.0) - baseline_metrics.get('ssl', 0.0)

        # Exact linear Shapley values (phi_j = w_j * (x_j - E[x_j]))
        phi = {
            'relevance': w_rel * diff_rel,
            'freshness': w_fresh * diff_fresh,
            'fatigue': -w_fatigue * diff_fatigue,
            'context': w_context * diff_context,
            'ssl': w_ssl * diff_ssl
        }

        total_diff = sum(phi.values())
        
        # Build natural-language explanation string
        parts = []
        sorted_phi = sorted(phi.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_phi:
            sign = "+" if val >= 0 else ""
            if name == 'relevance':
                parts.append(f"relevance contributes {sign}{val:.2f}")
            elif name == 'freshness':
                parts.append(f"freshness contributes {sign}{val:.2f}")
            elif name == 'fatigue':
                parts.append(f"fatigue penalty contributes {sign}{val:.2f}")
            elif name == 'ssl':
                parts.append(f"self-supervised representations contribute {sign}{val:.2f}")
            elif name == 'context':
                context_desc = "context match"
                if val > 0:
                    matched_contexts = []
                    if context_time:
                        matched_contexts.append(f"{context_time} time of day")
                    if context_device:
                        matched_contexts.append(f"{context_device} device")
                    if context_location:
                        matched_contexts.append(f"{context_location} location")
                    if matched_contexts:
                        context_desc = f"context match ({', '.join(matched_contexts)})"
                parts.append(f"{context_desc} contributes {sign}{val:.2f}")

        diff_sign = "+" if total_diff >= 0 else ""
        explanation_text = f"Scores {diff_sign}{total_diff:.2f} vs average because: {', '.join(parts)}."
        return total_diff, phi, explanation_text


class SASRecLimeExplainer:
    """Approximates feature importance for transformer sequence items via LIME perturbation models."""

    @staticmethod
    def explain_sequence(
        model: Any,
        user_history_indices: List[int],
        idx_to_item: Dict[int, str],
        item_details: Dict[str, dict],
        target_item_idx: int,
        num_perturbations: int = 15,
        drop_prob: float = 0.2
    ) -> str:
        """Approximate historical item contributions to the recommended target item's logits."""
        if not user_history_indices or target_item_idx == 0:
            return "Based on your recent sequence of interactions."

        history_len = len(user_history_indices)
        
        # Generate perturbations: binary matrix of shape (num_perturbations, history_len)
        np.random.seed(target_item_idx)
        X = np.random.binomial(1, 1 - drop_prob, size=(num_perturbations, history_len))
        X[0, :] = 1.0  # Keep original sequence intact as the first sample

        y = []
        model.eval()

        with torch.no_grad():
            for m in range(num_perturbations):
                # Construct perturbed sequence: keep only items where X[m, i] == 1
                perturbed_seq = [user_history_indices[i] for i in range(history_len) if X[m, i] == 1]
                if not perturbed_seq:
                    perturbed_seq = [0]
                
                # Pad sequence to max_seq_len
                pad_len = model.max_seq_len - len(perturbed_seq)
                padded = [0] * pad_len + perturbed_seq
                
                input_tensor = torch.tensor([padded], dtype=torch.long)
                logits, _, _ = model(input_tensor)
                
                # Retrieve log probs of target item at the last sequence step
                last_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(last_logits, dim=0)
                target_log_prob = log_probs[target_item_idx].item()
                y.append(target_log_prob)

        # Fit a simple Ridge regression: y = X * beta
        X_design = X.astype(np.float32)
        y_vec = np.array(y, dtype=np.float32)
        
        # Ridge normal equation: beta = (X^T X + lambda I)^-1 X^T y
        reg_lambda = 0.1
        try:
            beta = np.linalg.solve(X_design.T.dot(X_design) + reg_lambda * np.eye(history_len, dtype=np.float32), X_design.T.dot(y_vec))
        except Exception:
            beta = np.zeros(history_len)

        # Match beta coefficients to items in history
        item_contributions = []
        for i, idx in enumerate(user_history_indices):
            if idx > 0:
                item_id = idx_to_item.get(idx, f"item_{idx}")
                item_name = item_details.get(item_id, {}).get("title", item_id)
                item_contributions.append((item_name, beta[i]))

        # Sort by positive contribution
        item_contributions.sort(key=lambda x: x[1], reverse=True)
        top_contrib = [f"'{name}' (contributed +{val:.2f})" for name, val in item_contributions[:2] if val > 0.01]
        
        if top_contrib:
            return f"Recommended due to your recent interest in: {', '.join(top_contrib)}."
        return "Based on your recent sequence of interactions."
