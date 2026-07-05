"""Session builder for sequential recommendation.

Converts raw feedback data into ordered user interaction sequences
that can be fed into the SASRec sequential model.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

from data.feedback import get_all_feedback, get_user_feedback


def build_user_sequences(
    feedback_list: list,
    max_seq_len: int = 50
) -> Dict[str, List[str]]:
    """Group feedback by user, sort by timestamp, return ordered item sequences.

    Args:
        feedback_list: List of feedback dicts with 'user_id', 'item_id', 'timestamp'.
        max_seq_len: Maximum sequence length to keep (most recent items).

    Returns:
        Dictionary mapping user_id to list of item_ids in chronological order.
    """
    user_events = defaultdict(list)

    for fb in feedback_list:
        user_events[fb["user_id"]].append(
            (fb.get("timestamp", ""), fb["item_id"])
        )

    sequences = {}
    for user_id, events in user_events.items():
        # Sort by timestamp ascending (oldest first)
        events.sort(key=lambda x: x[0])
        # Keep only the item_ids, take the most recent max_seq_len
        item_ids = [item_id for _, item_id in events]
        sequences[user_id] = item_ids[-max_seq_len:]

    return sequences


def build_user_sequences_with_dwell(
    feedback_list: list,
    max_seq_len: int = 50
) -> Dict[str, List[dict]]:
    """Group feedback by user, sort by timestamp, return ordered item sequences with dwell times."""
    user_events = defaultdict(list)

    for fb in feedback_list:
        user_events[fb["user_id"]].append(
            (fb.get("timestamp", ""), fb["item_id"], fb.get("dwell_time", 0.0))
        )

    sequences = {}
    for user_id, events in user_events.items():
        # Sort by timestamp ascending
        events.sort(key=lambda x: x[0])
        # Keep only the item info
        sequences[user_id] = [
            {"item_id": item_id, "dwell_time": dwell_time}
            for _, item_id, dwell_time in events[-max_seq_len:]
        ]

    return sequences


def get_user_sequence(user_id: str, max_len: int = 50) -> List[str]:
    """Get a single user's interaction sequence from the database.

    Args:
        user_id: The user to fetch history for.
        max_len: Maximum number of recent items to return.

    Returns:
        List of item_ids in chronological order (oldest to newest).
    """
    feedback = get_user_feedback(user_id, limit=max_len)
    if not feedback:
        return []

    # get_user_feedback returns DESC order, reverse for chronological
    feedback.reverse()
    return [fb["item_id"] for fb in feedback]


def build_item_id_mapping(
    items: list,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build bidirectional mappings between item_id strings and integer indices.

    Index 0 is reserved for padding. Real items start at index 1.

    Args:
        items: List of item dicts (must have 'item_id' key) or list of item_id strings.

    Returns:
        Tuple of (item_to_idx, idx_to_item) dictionaries.
    """
    item_to_idx = {}
    idx_to_item = {0: "<PAD>"}

    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            item_id = item["item_id"]
        else:
            item_id = str(item)
        item_to_idx[item_id] = i
        idx_to_item[i] = item_id

    return item_to_idx, idx_to_item
