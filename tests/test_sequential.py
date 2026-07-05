"""Tests for Phase 3: SASRec Sequential Recommendation.

Tests session building, model forward pass, training loop,
and the sequential API endpoint.
"""

import os
import tempfile

import pytest
import torch

from sessions.session_builder import (
    build_user_sequences,
    get_user_sequence,
    build_item_id_mapping,
)
from models.sasrec import SASRec
from models.sequential_train import SequentialTrainer


# ---------------------------------------------------------------------------
# Session Builder Tests
# ---------------------------------------------------------------------------


class TestBuildUserSequences:
    """Tests for build_user_sequences()."""

    def test_basic_grouping(self):
        """Feedback is grouped by user and sorted by timestamp."""
        feedback = [
            {"user_id": "u1", "item_id": "i3", "timestamp": "2024-01-03"},
            {"user_id": "u1", "item_id": "i1", "timestamp": "2024-01-01"},
            {"user_id": "u2", "item_id": "i2", "timestamp": "2024-01-02"},
            {"user_id": "u1", "item_id": "i2", "timestamp": "2024-01-02"},
        ]
        seqs = build_user_sequences(feedback)
        assert seqs["u1"] == ["i1", "i2", "i3"]
        assert seqs["u2"] == ["i2"]

    def test_max_seq_len_truncation(self):
        """Sequences longer than max_seq_len are truncated to most recent."""
        feedback = [
            {"user_id": "u1", "item_id": f"i{i}", "timestamp": f"2024-01-{i:02d}"}
            for i in range(1, 11)
        ]
        seqs = build_user_sequences(feedback, max_seq_len=5)
        assert len(seqs["u1"]) == 5
        # Should keep the 5 most recent
        assert seqs["u1"] == ["i6", "i7", "i8", "i9", "i10"]

    def test_empty_feedback(self):
        """Empty feedback list returns empty dict."""
        seqs = build_user_sequences([])
        assert seqs == {}


class TestBuildItemIdMapping:
    """Tests for build_item_id_mapping()."""

    def test_mapping_with_dicts(self):
        """Item dicts are mapped correctly, with 0 reserved for padding."""
        items = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
        item_to_idx, idx_to_item = build_item_id_mapping(items)

        assert item_to_idx["a"] == 1
        assert item_to_idx["b"] == 2
        assert item_to_idx["c"] == 3
        assert idx_to_item[0] == "<PAD>"
        assert idx_to_item[1] == "a"

    def test_mapping_with_strings(self):
        """Plain string item_ids work too."""
        item_to_idx, idx_to_item = build_item_id_mapping(["x", "y"])
        assert item_to_idx["x"] == 1
        assert idx_to_item[2] == "y"

    def test_zero_reserved_for_padding(self):
        """Index 0 is always the padding token."""
        _, idx_to_item = build_item_id_mapping(["a"])
        assert 0 in idx_to_item
        assert idx_to_item[0] == "<PAD>"


class TestGetUserSequence:
    """Tests for get_user_sequence() with real DB."""

    def test_user_with_feedback(self):
        """User with feedback history returns chronological sequence."""
        from data.feedback import add_feedback

        add_feedback("u1", "item_a", "view", "2024-01-01T10:00:00")
        add_feedback("u1", "item_b", "click", "2024-01-01T11:00:00")
        add_feedback("u1", "item_c", "view", "2024-01-01T12:00:00")

        seq = get_user_sequence("u1")
        assert seq == ["item_a", "item_b", "item_c"]

    def test_user_without_feedback(self):
        """User with no history returns empty list."""
        seq = get_user_sequence("nonexistent_user")
        assert seq == []


# ---------------------------------------------------------------------------
# SASRec Model Tests
# ---------------------------------------------------------------------------


class TestSASRecModel:
    """Tests for the SASRec model architecture."""

    @pytest.fixture
    def model(self):
        """Create a small SASRec model for testing."""
        return SASRec(
            num_items=20,
            hidden_dim=32,
            max_seq_len=10,
            num_heads=2,
            num_blocks=1,
            dropout=0.0,
        )

    def test_forward_shape(self, model):
        """Forward pass produces correct output shape."""
        batch = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
        logits = model(batch)
        assert logits.shape == (1, 10, 20)

    def test_forward_batch(self, model):
        """Forward pass works with batch_size > 1."""
        batch = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            [5, 4, 3, 2, 1, 6, 7, 0, 0, 0],
        ])
        logits = model(batch)
        assert logits.shape == (2, 10, 20)

    def test_predict_next_returns_list(self, model):
        """predict_next returns a list of integer indices."""
        seq = [1, 2, 3]
        result = model.predict_next(seq, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(x, int) for x in result)

    def test_predict_next_excludes_padding(self, model):
        """Padding index 0 should never appear in predictions."""
        seq = [1, 2, 3]
        result = model.predict_next(seq, n=19)
        assert 0 not in result

    def test_predict_next_excludes_history(self, model):
        """Items already in the sequence should not be recommended."""
        seq = [1, 2, 3]
        result = model.predict_next(seq, n=19)
        for item in seq:
            assert item not in result

    def test_predict_next_empty_sequence(self, model):
        """Empty sequence returns empty list."""
        result = model.predict_next([], n=5)
        assert result == []

    def test_causal_mask_shape(self, model):
        """Causal mask has correct shape and structure."""
        mask = model._generate_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)
        assert mask.dtype == torch.bool
        # Diagonal and below should be False (allowed), upper triangle should be True (masked)
        assert mask[0, 0].item() is False
        assert mask[0, 1].item() is True


# ---------------------------------------------------------------------------
# Sequential Trainer Tests
# ---------------------------------------------------------------------------


class TestSequentialTrainer:
    """Tests for the SequentialTrainer class."""

    @pytest.fixture
    def small_setup(self):
        """Create a small model, trainer, and toy dataset."""
        items = [f"item_{i}" for i in range(10)]
        item_to_idx, idx_to_item = build_item_id_mapping(items)
        num_items = len(item_to_idx) + 1  # +1 for padding

        model = SASRec(
            num_items=num_items,
            hidden_dim=32,
            max_seq_len=10,
            num_heads=2,
            num_blocks=1,
            dropout=0.0,
        )
        trainer = SequentialTrainer(model, lr=1e-3)

        sequences = {
            "u1": ["item_0", "item_1", "item_2", "item_3", "item_4"],
            "u2": ["item_5", "item_6", "item_7"],
            "u3": ["item_2", "item_8", "item_9", "item_0"],
        }

        return trainer, sequences, item_to_idx, idx_to_item

    def test_prepare_training_data(self, small_setup):
        """Training data preparation produces valid tensors."""
        trainer, sequences, item_to_idx, _ = small_setup
        inputs, targets = trainer.prepare_training_data(
            sequences, item_to_idx, max_seq_len=10
        )

        assert inputs.shape[1] == 10
        assert targets.shape[1] == 10
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[0] > 0

    def test_prepare_training_data_empty(self, small_setup):
        """Empty sequences produce empty tensors."""
        trainer, _, item_to_idx, _ = small_setup
        inputs, targets = trainer.prepare_training_data(
            {}, item_to_idx, max_seq_len=10
        )
        assert inputs.shape[0] == 0

    def test_train_returns_losses(self, small_setup):
        """Training returns a list of loss values, one per epoch."""
        trainer, sequences, item_to_idx, _ = small_setup
        losses = trainer.train(
            sequences, item_to_idx, epochs=3, batch_size=4
        )
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)
        assert all(l > 0 for l in losses)

    def test_loss_decreases(self, small_setup):
        """Loss should generally decrease over training."""
        trainer, sequences, item_to_idx, _ = small_setup
        losses = trainer.train(
            sequences, item_to_idx, epochs=10, batch_size=4
        )
        # The average of the last 3 losses should be lower than the first
        assert sum(losses[-3:]) / 3 < losses[0]

    def test_save_and_load_model(self, small_setup):
        """Model can be saved and loaded with matching outputs."""
        trainer, sequences, item_to_idx, _ = small_setup
        trainer.train(sequences, item_to_idx, epochs=2, batch_size=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            trainer.save_model(path)
            assert os.path.exists(path)

            loaded_model, _ = SequentialTrainer.load_model(path)
            assert loaded_model is not None
            assert loaded_model.num_items == trainer.model.num_items


            # Both models should produce the same output for the same input
            test_seq = [1, 2, 3]
            orig_preds = trainer.model.predict_next(test_seq, n=5)
            loaded_preds = loaded_model.predict_next(test_seq, n=5)
            assert orig_preds == loaded_preds
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Sequential API Tests
# ---------------------------------------------------------------------------


class TestSequentialAPI:
    """Tests for the /sequential/{user_id} API endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client with the sequential router mounted."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.sequential import router
        from unittest.mock import MagicMock

        test_app = FastAPI()
        test_app.include_router(router)

        # Create a mock engine with default empty attributes
        mock_engine = MagicMock()
        mock_engine.seq_model = None
        mock_engine.seq_item_to_idx = None
        mock_engine.seq_idx_to_item = None
        mock_engine.cache.get.return_value = None
        test_app.state.engine = mock_engine


        with TestClient(test_app) as c:
            yield c

    def test_sequential_no_model(self, client):
        """Returns empty recommendations when no model is trained."""
        resp = client.get("/sequential/u1?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert isinstance(data["recommendations"], list)

    def test_sequential_no_history(self, client):
        """Returns empty recommendations for user with no history."""
        resp = client.get("/sequential/unknown_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []

    def test_sequential_with_trained_model(self, client):
        """Returns recommendations when model is trained and user has history."""
        from unittest.mock import MagicMock
        from data.feedback import add_feedback
        from data.items import add_item

        # Add items and feedback
        for i in range(10):
            add_item(f"item_{i}", f"Title {i}", f"tag{i}", "cat")
        for i in range(8):
            add_feedback("test_user", f"item_{i}", "view", f"2024-01-{i+1:02d}")

        # Train a small model
        items = [{"item_id": f"item_{i}"} for i in range(10)]
        item_to_idx, idx_to_item = build_item_id_mapping(items)
        num_items = len(item_to_idx) + 1

        model = SASRec(
            num_items=num_items, hidden_dim=32, max_seq_len=10,
            num_heads=2, num_blocks=1
        )
        trainer = SequentialTrainer(model, lr=1e-3)
        sequences = {"test_user": [f"item_{i}" for i in range(8)]}
        trainer.train(sequences, item_to_idx, epochs=5, batch_size=4)

        # Inject into mock engine
        client.app.state.engine.seq_model = model
        client.app.state.engine.seq_item_to_idx = item_to_idx
        client.app.state.engine.seq_idx_to_item = idx_to_item

        resp = client.get("/sequential/test_user?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test_user"
        # Should have some recommendations (model is trained)
        assert len(data["recommendations"]) > 0
        # Each rec should have item_id and score
        for rec in data["recommendations"]:
            assert "item_id" in rec
            assert "score" in rec
