"""Tests for model inference algorithms, multi-objective re-ranking, explainability, fairness, and causal estimators."""

import numpy as np
import torch
import pytest
from models.engine import get_freshness_score, get_context_match_score
from utils.diversity import mmr_rerank
from utils.explain import LinearShapleyExplainer, SASRecLimeExplainer
from utils.fairness import FairnessAuditor
from utils.surprise import SurpriseController
from utils.bandit import NeuralLinearBandit
from models.causal import PropensityEstimator
from models.lightgcn import LightGCN
from models.bcq import BCQQNetwork, BCQTrainer
from models.best_rec import BESTRec, BESTRecTrainer


class TestScoringAndContextMatch:
    def test_freshness_score(self):
        # Current year item
        item_2026 = {"metadata": {"year": 2026}}
        assert get_freshness_score(item_2026) == 1.0

        # 5 years old item
        item_2021 = {"metadata": {"year": 2021}}
        assert np.isclose(get_freshness_score(item_2021), 0.5, atol=1e-4)

        # 15 years old item -> clamped to 0.0
        item_old = {"metadata": {"year": 2010}}
        assert get_freshness_score(item_old) == 0.0

        # Missing metadata
        assert get_freshness_score({}) == 0.5

    def test_context_match_score(self):
        item_mobile = {"title": "Short Clip", "tags": "quick,short", "category": "videos"}
        # Match mobile + short
        score_mobile = get_context_match_score(item_mobile, device="mobile", location=None, time_of_day=None)
        assert score_mobile >= 0.5

        # Match evening movies
        item_movie = {"title": "Action Movie", "tags": "action", "category": "movies"}
        score_evening = get_context_match_score(item_movie, device=None, location=None, time_of_day="evening")
        assert score_evening >= 0.5

        # Match location in tags
        item_loc = {"title": "London Guide", "tags": "london,travel", "category": "articles"}
        score_loc = get_context_match_score(item_loc, device=None, location="london", time_of_day=None)
        assert score_loc == 1.0


class TestMMRDiversityReRanking:
    def test_mmr_pure_relevance_lambda_1(self):
        candidates = [
            {"item_id": "c1", "score": 0.9},
            {"item_id": "c2", "score": 0.8},
            {"item_id": "c3", "score": 0.7},
        ]
        item_map = {
            "c1": {"title": "Python 1", "tags": "python"},
            "c2": {"title": "Python 2", "tags": "python"},
            "c3": {"title": "Python 3", "tags": "python"},
        }
        re_ranked = mmr_rerank(candidates, item_map, n=2, diversity_lambda=1.0)
        assert len(re_ranked) == 2
        assert re_ranked[0]["item_id"] == "c1"
        assert re_ranked[1]["item_id"] == "c2"

    def test_mmr_diversity_lambda_0(self):
        candidates = [
            {"item_id": "c1", "score": 0.9},
            {"item_id": "c2", "score": 0.89},
            {"item_id": "c3", "score": 0.5},
        ]
        item_map = {
            "c1": {"title": "Machine Learning", "tags": "ai,python"},
            "c2": {"title": "Machine Learning Advanced", "tags": "ai,python"},
            "c3": {"title": "Italian Cuisine Recipe", "tags": "food,cooking"},
        }
        # With high diversity (lambda=0.0 or 0.1), distinct item c3 should be preferred over duplicate c2
        re_ranked = mmr_rerank(candidates, item_map, n=2, diversity_lambda=0.1)
        assert len(re_ranked) == 2
        assert re_ranked[0]["item_id"] == "c1"
        assert re_ranked[1]["item_id"] == "c3"


class TestExplainers:
    def test_linear_shapley_explainer(self):
        metrics = {"relevance": 0.9, "freshness": 0.8, "fatigue": 0.1, "context": 1.0, "ssl": 0.5}
        baseline = {"relevance": 0.5, "freshness": 0.5, "fatigue": 0.5, "context": 0.2, "ssl": 0.2}
        weights = {"w_relevance": 1.0, "w_freshness": 0.2, "w_fatigue": 0.3, "w_context": 0.4, "w_ssl": 0.15}

        total_diff, phi, explanation = LinearShapleyExplainer.explain_recommendation(
            "item_1", metrics, baseline, weights, context_device="mobile"
        )
        assert isinstance(total_diff, float)
        assert "relevance" in phi
        assert "freshness" in phi
        assert "fatigue" in phi
        assert "context" in phi
        assert "ssl" in phi
        assert len(explanation) > 10


class TestFairnessAndSurpriseControllers:
    def test_fairness_auditor_pid_and_di(self):
        auditor = FairnessAuditor(target_di=1.0, kp=0.5, ki=0.05, kd=0.1)
        # Empty impressions -> DI = 1.0
        assert auditor.compute_di() == 1.0

        # Add 10 majority impressions (non-books)
        auditor.record_recommendations([{"category": "movies"}] * 10)
        di_low = auditor.audit_and_update_pid()
        assert di_low < 1.0
        assert auditor.lambda_fair > 0.0  # Bonus should activate for minority group

        # Bonus applied to minority category
        score_minority = auditor.get_fairness_score("books", score_rrf=1.0)
        score_majority = auditor.get_fairness_score("movies", score_rrf=1.0)
        assert score_minority > score_majority

    def test_surprise_controller_dirichlet_and_kl(self):
        controller = SurpriseController(target_kl=0.5, kp=0.5)
        # Dirichlet prior with Laplace smoothing
        alpha = controller.get_user_dirichlet_prior([], {})
        assert len(alpha) == controller.num_cats
        assert all(a == 1.0 for a in alpha)

        # Equal distributions -> KL = 0.0
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        kl = controller.compute_kl_divergence(p, q)
        assert np.isclose(kl, 0.0, atol=1e-5)


class TestBanditAndCausalEstimator:
    def test_neural_linear_bandit_selection_and_update(self, clean_db):
        bandit = NeuralLinearBandit()
        context_vec = np.random.randn(73).astype(np.float32)

        arm_id, weights = bandit.select_action(context_vec)
        assert 0 <= arm_id < bandit.num_actions
        assert len(weights) == 6

        # Update with reward
        bandit.update(arm_id, context_vec, reward=1.0)
        assert np.any(bandit.f[arm_id] != 0.0)

    def test_propensity_estimator_and_ips_weight(self):
        estimator = PropensityEstimator()
        # Predict propensity probability
        prob = estimator.predict_propensity(cohort="A", device="mobile", time_of_day="evening", category="movies")
        assert 0.05 <= prob <= 0.95

        # IPS weight calculation for clicked item (1 / p)
        weight_clicked = estimator.get_ips_weight(cohort="A", device="mobile", time_of_day="evening", category="movies", clicked=True)
        assert 0.1 <= weight_clicked <= 10.0

        # IPS weight for non-clicked item (1 / (1-p))
        weight_non_clicked = estimator.get_ips_weight(cohort="A", device="mobile", time_of_day="evening", category="movies", clicked=False)
        assert 0.1 <= weight_non_clicked <= 10.0


class TestDeepNeuralModels:
    def test_lightgcn_forward_and_loss(self):
        model = LightGCN(num_users=10, num_items=15, emb_dim=32, num_layers=2)
        interactions = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5)]
        L = model.get_adj_matrix(interactions)
        assert L.shape == (25, 25)

        users_emb, items_emb = model(L)
        assert users_emb.shape == (10, 32)
        assert items_emb.shape == (15, 32)

        loss = model.bpr_loss(users_emb, items_emb, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([5, 6]))
        assert loss.item() > 0.0

    def test_bcq_q_network_forward(self):
        bcq = BCQQNetwork(state_dim=390, item_emb_dim=384, latent_dim=64)
        state = torch.randn(4, 390)
        item_emb = torch.randn(4, 384)
        q_values = bcq(state, item_emb)
        assert q_values.shape == (4,)

        # Multiple candidates per state: (batch_size, num_cand, item_emb_dim)
        item_emb_candidates = torch.randn(4, 10, 384)
        q_cand_values = bcq(state, item_emb_candidates)
        assert q_cand_values.shape == (4, 10)

    def test_best_rec_forward(self):
        best_rec = BESTRec(num_items=50, hidden_dim=64, max_seq_len=20)
        seq_input = torch.randint(0, 50, (4, 20))
        user_emb, out, nap_logits = best_rec(seq_input)
        assert user_emb.shape == (4, 256)
        assert out.shape == (4, 20, 64)
        assert nap_logits.shape == (4, 5)
