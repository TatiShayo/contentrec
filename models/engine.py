import os
import pickle
import threading
import numpy as np
import scipy.sparse as sp
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False

from data.feedback import get_all_feedback
from data.items import get_all_items, search_by_tags
import config

class RecommendationEngine:
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.cold_start_threshold = config.COLD_START_THRESHOLD
        self._lock = threading.Lock()
        if LIGHTFM_AVAILABLE:
            self.model = LightFM(loss='warp')
            self.dataset = Dataset()
        else:
            self.model = None
            self.dataset = None
            
        self.user_id_map = {}
        self.item_id_map = {}
        self.item_features = None
        self.interactions = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and LIGHTFM_AVAILABLE:
            try:
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
                    self.model = state['model']
                    self.dataset = state['dataset']
                    self.user_id_map = state['user_id_map']
                    self.item_id_map = state['item_id_map']
                    self.item_features = state['item_features']
                    self.interactions = state['interactions']
            except Exception as e:
                print(f"Error loading model: {e}")

    def _save_model(self):
        if not LIGHTFM_AVAILABLE:
            return
            
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            state = {
                'model': self.model,
                'dataset': self.dataset,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'item_features': self.item_features,
                'interactions': self.interactions
            }
            pickle.dump(state, f)

    def train(self):
        with self._lock:
            if not LIGHTFM_AVAILABLE:
                print("LightFM not available, skipping collaborative filtering training.")
                return

            feedback = get_all_feedback()
            items = get_all_items()

            if not feedback or not items:
                return

            user_ids = list(set(f['user_id'] for f in feedback))
            item_ids = [i['item_id'] for i in items]

            # Build item features
            all_tags = set()
            for item in items:
                if item['tags']:
                    all_tags.update([t.strip() for t in item['tags'].split(',')])

            self.dataset.fit(
                users=user_ids,
                items=item_ids,
                item_features=list(all_tags)
            )

            (self.interactions, weights) = self.dataset.build_interactions(
                [(f['user_id'], f['item_id']) for f in feedback]
            )

            item_features_list = []
            for item in items:
                tags = [t.strip() for t in item['tags'].split(',')] if item['tags'] else []
                item_features_list.append((item['item_id'], tags))

            self.item_features = self.dataset.build_item_features(item_features_list)

            self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()

            self.model.fit(
                self.interactions,
                item_features=self.item_features,
                epochs=30,
                num_threads=4
            )

            self._save_model()

    def partial_fit(self, new_feedback):
        with self._lock:
            if not LIGHTFM_AVAILABLE:
                return

            if self.interactions is not None:
                 (new_interactions, _) = self.dataset.build_interactions(
                     [(f['user_id'], f['item_id']) for f in new_feedback]
                 )
                 self.model.fit_partial(new_interactions, item_features=self.item_features)
                 self._save_model()
            else:
                self.train()

    def recommend(self, user_id, n=10, features=None):
        with self._lock:
            feedback = [f for f in get_all_feedback() if f['user_id'] == user_id]

            if len(feedback) < self.cold_start_threshold or not LIGHTFM_AVAILABLE or user_id not in self.user_id_map:
                return self.cold_start_recommend(tags_or_category=features, n=n)

            user_idx = self.user_id_map[user_id]
            n_items = len(self.item_id_map)

            scores = self.model.predict(
                user_idx,
                np.arange(n_items),
                item_features=self.item_features
            )

            top_items_idx = np.argsort(-scores)[:n]

            inv_item_map = {v: k for k, v in self.item_id_map.items()}

            recs = []
            for idx in top_items_idx:
                recs.append({
                    "item_id": inv_item_map[idx],
                    "score": float(scores[idx])
                })

            return recs

    def similar_items(self, item_id, n=5):
        with self._lock:
            if not LIGHTFM_AVAILABLE or item_id not in self.item_id_map:
                return []

            item_idx = self.item_id_map[item_id]

            # Get item representations
            _, item_embeddings = self.model.get_item_representations(self.item_features)

            target_embedding = item_embeddings[item_idx]

            # Cosine similarity
            scores = np.dot(item_embeddings, target_embedding) / (
                np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(target_embedding) + 1e-10
            )

            top_items_idx = np.argsort(-scores)[1:n+1] # Skip itself

            inv_item_map = {v: k for k, v in self.item_id_map.items()}

            recs = []
            for idx in top_items_idx:
                recs.append({
                    "item_id": inv_item_map[idx],
                    "score": float(scores[idx])
                })

            return recs

    def cold_start_recommend(self, tags_or_category=None, n=10):
        items = []
        if tags_or_category:
            if isinstance(tags_or_category, str):
                tags = [t.strip() for t in tags_or_category.split(',')]
            else:
                tags = tags_or_category
                
            for tag in tags:
                items.extend(search_by_tags(tag))
        
        # Fallback to popularity
        if not items:
            feedback = get_all_feedback()
            item_counts = {}
            for f in feedback:
                item_counts[f['item_id']] = item_counts.get(f['item_id'], 0) + 1
            
            sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
            return [{"item_id": i[0], "score": float(i[1])} for i in sorted_items[:n]]
            
        # Unique items from tag search
        seen = set()
        unique_items = []
        for item in items:
            if item['item_id'] not in seen:
                unique_items.append({"item_id": item['item_id'], "score": 1.0})
                seen.add(item['item_id'])
                
        return unique_items[:n]
