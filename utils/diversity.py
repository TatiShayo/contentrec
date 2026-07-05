import numpy as np
from typing import List, Dict, Any
from embeddings.text import TextEmbedder


def mmr_rerank(
    candidates: List[Dict[str, Any]],
    item_details_map: Dict[str, Dict[str, Any]],
    n: int = 10,
    diversity_lambda: float = 0.5,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance (MMR) re-ranking.

    Balances relevance (relevance scores from recommendation algorithms)
    with diversity (calculated using cosine similarity on SBERT embeddings).

    Args:
        candidates: Candidate recommendation list of dicts with 'item_id' and
          'score'.
        item_details_map: Dict mapping item_id to details (title, tags, category).
        n: Max number of items to return after re-ranking.
        diversity_lambda: Balance factor [0.0, 1.0]. 1.0 means pure relevance,
          0.0 means pure diversity.

    Returns:
        List of re-ranked recommendations.
    """
    if diversity_lambda >= 1.0 or len(candidates) <= 1 or n <= 1:
        return candidates[:n]

    embedder = TextEmbedder()

    # Get embeddings for all candidates
    item_embeddings = {}
    for c in candidates:
        iid = c["item_id"]
        details = item_details_map.get(iid)
        if details:
            try:
                item_embeddings[iid] = embedder.embed_item(details)
            except Exception:
                item_embeddings[iid] = np.zeros(384, dtype=np.float32)
        else:
            item_embeddings[iid] = np.zeros(384, dtype=np.float32)

    # Normalize scores to [0.0, 1.0] for fair comparison with similarity
    scores = np.array([c["score"] for c in candidates])
    min_s = float(scores.min())
    max_s = float(scores.max())
    score_range = max_s - min_s

    if score_range > 0:
        normalized_scores = {
            c["item_id"]: (c["score"] - min_s) / score_range for c in candidates
        }
    else:
        normalized_scores = {c["item_id"]: 1.0 for c in candidates}

    # Start selection with the highest scored item
    selected = []
    unselected = candidates.copy()
    first = unselected.pop(0)
    selected.append(first)

    while len(selected) < n and unselected:
        best_mmr = -float("inf")
        best_idx = -1

        for idx, cand in enumerate(unselected):
            iid = cand["item_id"]
            relevance = normalized_scores[iid]
            cand_emb = item_embeddings[iid]

            # Find maximum similarity to any currently selected items
            max_similarity = -1.0
            for sel in selected:
                sel_emb = item_embeddings[sel["item_id"]]
                # Dot product is cosine similarity as embeddings are L2 normalized
                similarity = float(np.dot(cand_emb, sel_emb))
                if similarity > max_similarity:
                    max_similarity = similarity

            # MMR formula
            mmr_score = diversity_lambda * relevance - (1.0 - diversity_lambda) * max_similarity

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected.append(unselected.pop(best_idx))
        else:
            break

    return selected
