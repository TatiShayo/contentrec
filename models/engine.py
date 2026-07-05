import os
import time
import pickle
import platform
import logging
import threading
import numpy as np
import scipy.sparse as sp

try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    # LightFM Cython extensions crash with access violation on Windows
    # (known bug in _run_epoch). Disable on Windows.
    if platform.system() == 'Windows':
        print("WARNING: LightFM disabled on Windows (Cython crash). Using content-based fallback.")
        LIGHTFM_AVAILABLE = False
    else:
        LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False

try:
    import faiss
    from search.faiss_index import FAISSIndex
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from models.sasrec import SASRec
    from models.sequential_train import SequentialTrainer
    SASREC_AVAILABLE = True
except ImportError:
    SASREC_AVAILABLE = False

from data.feedback import get_all_feedback
from data.items import get_all_items, search_by_tags, get_items_by_ids
import config
from utils.cache import RecommendationCache
from utils.metrics import MetricsTracker
from utils.diversity import mmr_rerank
from utils.ab_test import ABTestManager

from utils.bandit import NeuralLinearBandit
from utils.fairness import FairnessAuditor
from utils.explain import LinearShapleyExplainer, SASRecLimeExplainer
from models.bcq import BCQQNetwork, BCQTrainer

from models.causal import PropensityEstimator
from models.best_rec import BESTRec, BESTRecTrainer
from utils.surprise import SurpriseController
from data.database import add_impression


def get_freshness_score(item) -> float:
    metadata = item.get("metadata") or {}
    year = metadata.get("published_year") or metadata.get("year")
    if year:
        try:
            diff = 2026 - int(year)
            return max(0.0, 1.0 - (diff * 0.1))
        except:
            pass
    return 0.5


def get_context_match_score(item, device, location, time_of_day) -> float:
    score = 0.0
    tags = (item.get("tags") or "").lower()
    category = (item.get("category") or "").lower()
    
    if device:
        device = device.lower()
        if device == "mobile" and ("short" in tags or "quick" in tags):
            score += 0.5
        elif device == "desktop" and ("detailed" in tags or "deep" in tags):
            score += 0.5
            
    if time_of_day:
        time_of_day = time_of_day.lower()
        if time_of_day in ["evening", "night"] and category in ["movies", "entertainment"]:
            score += 0.5
        elif time_of_day in ["morning", "afternoon"] and category in ["articles", "news", "education", "books"]:
            score += 0.5
            
    if location:
        location = location.lower()
        if location in tags or location in category:
            score += 1.0
            
    return min(1.0, score)


class RecommendationEngine:
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.cold_start_threshold = config.COLD_START_THRESHOLD
        self._lock = threading.Lock()
        
        # Collaborative Filtering (LightFM)
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
        
        # Semantic Search (FAISS)
        self.faiss_index = None
        
        # Sequential Model (SASRec)
        self.seq_model = None
        self.seq_item_to_idx = {}
        self.seq_idx_to_item = {}

        # Cache & Metrics Managers
        self.cache = RecommendationCache()
        self.metrics = MetricsTracker()

        # Retraining scheduler state
        self.last_trained_time = time.time()
        self.new_feedback_counter = 0

        # Graph Collaborative Filtering (LightGCN)
        self.lightgcn_trainer = None
        self.lightgcn_user_emb = None
        self.lightgcn_item_emb = None

        # Neural Linear Bandit for weight auto-tuning
        self.bandit = NeuralLinearBandit()

        # Fairness Auditor
        self.fairness_auditor = FairnessAuditor()

        # BCQ RL Q-network
        self.bcq_model = BCQQNetwork()
        self.bcq_trainer = BCQTrainer(self.bcq_model)

        # Propensity Estimator
        self.propensity_estimator = PropensityEstimator()

        # Surprise controller
        self.surprise_controller = SurpriseController()

        # BEST-Rec self-supervised sequence model
        self.best_rec_model = None
        self.best_rec_trainer = None

        # Load models on startup
        self._load_model()
        self._load_faiss_index()
        self._load_sequential_model()
        self._load_lightgcn()
        self._load_bcq()
        self._load_best_rec()

        # Online incremental training loop for SASRec
        self.seq_online_trainer = None
        self.seq_online_stop_event = threading.Event()
        self.seq_online_thread = None
        
        if SASREC_AVAILABLE and self.seq_model is not None:
            from models.sequential_train import OnlineSequentialTrainer, run_online_sequential_learning
            import copy
            self.seq_model_train = copy.deepcopy(self.seq_model)
            self.seq_online_trainer = OnlineSequentialTrainer(
                model_train=self.seq_model_train,
                model_serve=self.seq_model,
                lr=1e-4
            )
            self.seq_online_thread = threading.Thread(
                target=run_online_sequential_learning,
                args=(self.seq_online_trainer, self.seq_online_stop_event),
                daemon=True
            )
            self.seq_online_thread.start()

    def _load_model(self):
        """Load collaborative filtering model."""
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
                logging.info("Loaded LightFM model successfully.")
            except Exception as e:
                logging.error(f"Error loading LightFM model: {e}")

    def _save_model(self):
        """Save collaborative filtering model."""
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

    def _load_faiss_index(self):
        """Load FAISS index."""
        if not FAISS_AVAILABLE:
            return
        try:
            if os.path.exists(config.FAISS_INDEX_PATH):
                self.faiss_index = FAISSIndex()
                self.faiss_index.load(config.FAISS_INDEX_PATH)
                logging.info("Loaded FAISS index successfully.")
            else:
                logging.info("FAISS index file not found. Rebuilding...")
                self.rebuild_faiss_index()
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")

    def rebuild_faiss_index(self):
        """Rebuild FAISS index from all items currently in database."""
        if not FAISS_AVAILABLE:
            return
        try:
            items = get_all_items()
            if items:
                self.faiss_index = FAISSIndex()
                self.faiss_index.build_index(items)
                self.faiss_index.save(config.FAISS_INDEX_PATH)
                logging.info("Built and saved FAISS index successfully.")
        except Exception as e:
            logging.error(f"Error building FAISS index: {e}")

    def _load_sequential_model(self):
        """Load SASRec model and its item mapping dictionary."""
        if not SASREC_AVAILABLE:
            return
        try:
            if os.path.exists(config.SASREC_MODEL_PATH) and os.path.exists(config.SASREC_MAP_PATH):
                with open(config.SASREC_MAP_PATH, 'rb') as f:
                    maps = pickle.load(f)
                    self.seq_item_to_idx = maps['item_to_idx']
                    self.seq_idx_to_item = maps['idx_to_item']
                self.seq_model, _ = SequentialTrainer.load_model(
                    config.SASREC_MODEL_PATH,
                    num_items=len(self.seq_item_to_idx) + 1
                )
                logging.info("Loaded SASRec model successfully.")
            else:
                logging.info("SASRec model or mappings not found.")
        except Exception as e:
            logging.error(f"Error loading SASRec model: {e}")

    def _load_lightgcn(self):
        """Load LightGCN model and embeddings."""
        try:
            from models.lightgcn import LightGCNTrainer
            self.lightgcn_trainer = LightGCNTrainer()
            if self.lightgcn_trainer.load(config.LIGHTGCN_MODEL_PATH):
                self.lightgcn_user_emb = self.lightgcn_trainer.users_emb
                self.lightgcn_item_emb = self.lightgcn_trainer.items_emb
                logging.info("Loaded LightGCN model successfully.")
            else:
                logging.info("LightGCN model file not found.")
        except Exception as e:
            logging.error(f"Error loading LightGCN: {e}")

    def _load_bcq(self):
        """Load BCQ model weights."""
        try:
            if os.path.exists(config.BCQ_MODEL_PATH):
                if self.bcq_trainer.load(config.BCQ_MODEL_PATH):
                    logging.info("Loaded BCQ Q-network successfully.")
            else:
                logging.info("BCQ model file not found.")
        except Exception as e:
            logging.error(f"Error loading BCQ: {e}")

    def _load_best_rec(self):
        """Load BEST-Rec model and weights."""
        try:
            from models.best_rec import BESTRec, BESTRecTrainer
            num_items = len(self.seq_item_to_idx) + 1 if self.seq_item_to_idx else 1000
            self.best_rec_model = BESTRec(num_items=num_items)
            self.best_rec_trainer = BESTRecTrainer(self.best_rec_model)
            if os.path.exists("data/best_rec.pth"):
                self.best_rec_trainer.load("data/best_rec.pth")
                logging.info("Loaded BEST-Rec model successfully.")
            else:
                logging.info("BEST-Rec model file not found.")
        except Exception as e:
            logging.error(f"Error loading BEST-Rec model: {e}")

    def _get_context_vector(self, user_id, device, time_of_day) -> np.ndarray:
        """Construct a 73-dimensional context vector for the Neural Linear Bandit."""
        # 1. GCN user embedding (64)
        gcn_emb = np.zeros(64, dtype=np.float32)
        if (self.lightgcn_trainer is not None and 
            self.lightgcn_user_emb is not None and 
            user_id in self.lightgcn_trainer.user_to_idx):
            idx = self.lightgcn_trainer.user_to_idx[user_id]
            gcn_emb = self.lightgcn_user_emb[idx]

        # 2. Time of day one-hot (4)
        tod_vec = np.zeros(4, dtype=np.float32)
        tod = (time_of_day or "").lower()
        if tod == "morning":
            tod_vec[0] = 1.0
        elif tod == "afternoon":
            tod_vec[1] = 1.0
        elif tod == "evening":
            tod_vec[2] = 1.0
        elif tod == "night":
            tod_vec[3] = 1.0

        # 3. Device one-hot (3)
        dev_vec = np.zeros(3, dtype=np.float32)
        dev = (device or "").lower()
        if dev == "mobile":
            dev_vec[0] = 1.0
        elif dev == "desktop":
            dev_vec[1] = 1.0
        elif dev == "tablet":
            dev_vec[2] = 1.0

        # 4. History size (1)
        fb_count = 0.0
        try:
            from data.feedback import get_user_feedback
            fb_count = float(len(get_user_feedback(user_id, limit=100)))
        except:
            pass

        # 5. Historical CTR (1)
        ctr = 0.0
        with self.metrics._lock:
            user_served = sum(1 for k, v in self.metrics.served_recs.items() if k[0] == user_id)
            if user_served > 0:
                ctr = min(1.0, fb_count / user_served)

        return np.concatenate([gcn_emb, tod_vec, dev_vec, [fb_count], [ctr]])

    def _get_bcq_state(self, user_id, user_feedback) -> np.ndarray:
        """Construct a 390-dimensional state vector for the BCQ Q-network."""
        user_history_items = [f["item_id"] for f in user_feedback[-10:]]

        # 1. Mean SBERT embedding (384)
        state_emb = np.zeros(384, dtype=np.float32)
        if user_history_items and self.faiss_index is not None:
            try:
                history_details = get_items_by_ids(user_history_items)
                embs = []
                for iid in user_history_items:
                    det = history_details.get(iid)
                    if det:
                        embs.append(self.faiss_index._embedder.embed_item(det))
                if embs:
                    state_emb = np.mean(embs, axis=0)
            except:
                pass

        # 2. Fatigue counts (5)
        categories = ["movies", "music", "books", "articles", "news"]
        category_counts = {}
        for iid in user_history_items:
            try:
                from data.items import get_item
                item = get_item(iid)
                if item:
                    cat = (item.get("category") or "").lower()
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            except:
                pass

        fatigue_vec = np.zeros(len(categories), dtype=np.float32)
        for cat_idx, cat_name in enumerate(categories):
            fatigue_vec[cat_idx] = float(category_counts.get(cat_name, 0))

        # 3. Session step (1)
        step_idx = np.array([float(len(user_history_items))], dtype=np.float32)

        return np.concatenate([state_emb, fatigue_vec, step_idx])

    def get_lightgcn_recs(self, user_id, n=10):
        """Get LightGCN collaborative filtering recommendations for a user."""
        if self.lightgcn_trainer is None or self.lightgcn_user_emb is None or self.lightgcn_item_emb is None:
            return []
        try:
            if user_id not in self.lightgcn_trainer.user_to_idx:
                # Cold-start user: use average user embedding
                u_vector = np.mean(self.lightgcn_user_emb, axis=0)
            else:
                u_idx = self.lightgcn_trainer.user_to_idx[user_id]
                u_vector = self.lightgcn_user_emb[u_idx]
            
            # Compute dot product scores against all item embeddings
            scores = np.dot(self.lightgcn_item_emb, u_vector)
            top_indices = np.argsort(-scores)[:n]
            
            recs = []
            for rank, idx in enumerate(top_indices):
                item_id = self.lightgcn_trainer.idx_to_item.get(idx)
                if item_id:
                    recs.append({
                        "item_id": item_id,
                        "score": float(scores[idx]),
                        "source": "collaborative_gcn",
                        "explanation": "Recommended based on graph collaborative filtering preferences."
                    })
            return recs
        except Exception as e:
            logging.error(f"Error getting LightGCN recommendations: {e}")
            return []

    def train_sequential_model(self):
        """Train SASRec sequential model from feedback/sequences."""
        if not SASREC_AVAILABLE:
            return
        try:
            from data.feedback import get_all_feedback
            from sessions.session_builder import build_user_sequences_with_dwell, build_item_id_mapping
            
            feedback = get_all_feedback()
            items = get_all_items()
            
            if not feedback or not items:
                logging.info("Insufficient data to train sequential model.")
                return
                
            # Build item mappings
            self.seq_item_to_idx, self.seq_idx_to_item = build_item_id_mapping(items)
            num_items = len(self.seq_item_to_idx) + 1  # 0 is padding
            
            # Build user sequences with dwell times
            sequences = build_user_sequences_with_dwell(feedback, max_seq_len=50)
            
            # Initialize SASRec model
            self.seq_model = SASRec(
                num_items=num_items,
                hidden_dim=64,
                max_seq_len=50,
                num_heads=2,
                num_blocks=2,
                dropout=0.2
            )
            
            trainer = SequentialTrainer(self.seq_model, lr=1e-3, device='cpu', propensity_estimator=self.propensity_estimator)
            logging.info("Training SASRec sequential model...")
            trainer.train(sequences, self.seq_item_to_idx, epochs=10, batch_size=32)
            
            # Save model and mappings
            trainer.save_model(config.SASREC_MODEL_PATH)
            with open(config.SASREC_MAP_PATH, 'wb') as f:
                pickle.dump({
                    'item_to_idx': self.seq_item_to_idx,
                    'idx_to_item': self.seq_idx_to_item
                }, f)
            logging.info("Saved SASRec model and mappings successfully.")
        except Exception as e:
            logging.error(f"Error training SASRec model: {e}")

    def train(self):
        """Train all recommendation models and invalidate cache."""
        with self._lock:
            # Reset retraining counters
            self.new_feedback_counter = 0
            self.last_trained_time = time.time()

            # Clear cache upon model retraining
            self.cache.clear()
            
            # Refresh fairness catalogue counts
            self.fairness_auditor.refresh_catalog_counts()

            # 0. Causal Propensity Model
            try:
                logging.info("Training Propensity Estimator...")
                self.propensity_estimator.train_model()
            except Exception as e:
                logging.error(f"Error training propensity model: {e}")
            
            # 1. Collaborative Filtering (LightFM)
            if LIGHTFM_AVAILABLE:
                try:
                    feedback = get_all_feedback()
                    items = get_all_items()

                    if feedback and items:
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

                        n_threads = 1 if platform.system() == 'Windows' else 4
                        self.model.fit(
                            self.interactions,
                            item_features=self.item_features,
                            epochs=30,
                            num_threads=n_threads
                        )

                        self._save_model()
                        logging.info("LightFM model trained and saved.")
                except Exception as e:
                    logging.error(f"Error training LightFM model: {e}")
            else:
                logging.info("LightFM not available, skipping collaborative filtering training.")

            # 2. Semantic Search (FAISS)
            if FAISS_AVAILABLE:
                self.rebuild_faiss_index()

            # 3. Sequential Model (SASRec)
            if SASREC_AVAILABLE:
                self.train_sequential_model()

            # 4. Graph Collaborative Filtering (LightGCN)
            try:
                from models.lightgcn import LightGCNTrainer
                trainer = LightGCNTrainer(emb_dim=64, epochs=20)
                feedback = get_all_feedback()
                items = get_all_items()
                if feedback and items:
                    user_emb, item_emb = trainer.train_model(
                        feedback, items, self.faiss_index._embedder if (self.faiss_index is not None) else None
                    )
                    if user_emb is not None:
                        self.lightgcn_user_emb = user_emb
                        self.lightgcn_item_emb = item_emb
                        self.lightgcn_trainer = trainer
                        trainer.save(config.LIGHTGCN_MODEL_PATH, user_emb, item_emb)
                        logging.info("LightGCN trained and saved successfully.")
            except Exception as e:
                logging.error(f"Error training LightGCN: {e}")

            # 5. Reinforcement Learning (BCQ Q-Network)
            try:
                feedback = get_all_feedback()
                items = get_all_items()
                if feedback and items:
                    logging.info("Training BCQ Q-network...")
                    self.bcq_trainer.train(feedback, items, self.faiss_index._embedder if (self.faiss_index is not None) else None, epochs=10)
                    self.bcq_trainer.save(config.BCQ_MODEL_PATH)
                    logging.info("BCQ Q-network trained and saved successfully.")
            except Exception as e:
                logging.error(f"Error training BCQ: {e}")

            # 6. Self-Supervised Universal User Representations (BEST-Rec)
            try:
                from models.best_rec import BESTRec, BESTRecTrainer
                from sessions.session_builder import build_user_sequences_with_dwell
                feedback = get_all_feedback()
                items = get_all_items()
                if feedback and items:
                    logging.info("Training BEST-Rec model...")
                    item_ids = sorted([i['item_id'] for i in items])
                    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
                    
                    categories_list = ["movies", "music", "books", "articles", "news"]
                    cat_to_idx = {cat: idx for idx, cat in enumerate(categories_list)}
                    
                    item_to_cat_idx = {}
                    for item in items:
                        iid = item["item_id"]
                        cat = (item.get("category") or "").lower()
                        cat_idx = cat_to_idx.get(cat, 0)
                        if iid in item_to_idx:
                            item_to_cat_idx[item_to_idx[iid]] = cat_idx
                            
                    sequences_dict = build_user_sequences_with_dwell(feedback, max_seq_len=30)
                    seq_list = []
                    for uid, events in sequences_dict.items():
                        if not events:
                            continue
                        if isinstance(events[0], dict):
                            seq_indices = [item_to_idx[e['item_id']] for e in events if e['item_id'] in item_to_idx]
                        else:
                            seq_indices = [item_to_idx[iid] for iid in events if iid in item_to_idx]
                        if len(seq_indices) >= 2:
                            seq_list.append(seq_indices)
                            
                    num_items = len(item_to_idx) + 1
                    self.best_rec_model = BESTRec(num_items=num_items)
                    self.best_rec_trainer = BESTRecTrainer(self.best_rec_model)
                    
                    self.best_rec_trainer.train_epoch(seq_list, item_to_cat_idx)
                    self.best_rec_trainer.save("data/best_rec.pth")
                    logging.info("BEST-Rec model trained and saved successfully.")
            except Exception as e:
                logging.error(f"Error training BEST-Rec: {e}")

    def partial_fit(self, new_feedback):
        """Incorporate new feedback on the fly (LightFM only)."""
        with self._lock:
            # Invalidate cached items for users receiving new feedback
            for fb in new_feedback:
                self.cache.invalidate_user(fb['user_id'])
                
            if not LIGHTFM_AVAILABLE:
                return

            if self.interactions is not None:
                (new_interactions, _) = self.dataset.build_interactions(
                    [(f['user_id'], f['item_id']) for f in new_feedback]
                )
                n_threads = 1 if platform.system() == 'Windows' else 4
                self.model.fit_partial(new_interactions, item_features=self.item_features, num_threads=n_threads)
                self._save_model()
            else:
                self.train()

    def get_sequential_recs(self, user_id, n=10):
        """Get sequential next-item recommendations using SASRec."""
        if not SASREC_AVAILABLE or self.seq_model is None or not self.seq_item_to_idx:
            return []
        try:
            from sessions.session_builder import get_user_sequence
            sequence = get_user_sequence(user_id, max_len=50)
            if not sequence:
                return []
            
            idx_seq = [self.seq_item_to_idx.get(iid, 0) for iid in sequence]
            idx_seq = [i for i in idx_seq if i != 0]
            if not idx_seq:
                return []
                
            top_indices = self.seq_model.predict_next(idx_seq, n=n)
            recs = []
            for rank, idx in enumerate(top_indices):
                item_id = self.seq_idx_to_item.get(idx, None)
                if item_id:
                    recs.append({
                        "item_id": item_id,
                        "score": float(1.0 - rank * 0.05),
                        "source": "sequential",
                        "explanation": "Based on your recent interaction sequence."
                    })
            return recs
        except Exception as e:
            logging.error(f"Error getting sequential recommendations: {e}")
            return []

    def get_content_similarity_recs(self, user_id, n=10):
        """Get content similarity recommendations using FAISS based on the user's latest item."""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return []
        try:
            from sessions.session_builder import get_user_sequence
            sequence = get_user_sequence(user_id, max_len=5)
            if not sequence:
                return []
            
            last_item_id = sequence[-1]
            from data.items import get_item
            item_details = get_item(last_item_id)
            if not item_details:
                return []
                
            embedding = self.faiss_index._embedder.embed_item(item_details)
            raw_recs = self.faiss_index.search(embedding, n=n + 1)
            
            recs = []
            for r in raw_recs:
                if r['item_id'] != last_item_id:
                    recs.append({
                        "item_id": r['item_id'],
                        "score": r['score'],
                        "source": "content",
                        "explanation": f"Similar to '{item_details.get('title', 'interacted item')}' from your history."
                    })
            return recs[:n]
        except Exception as e:
            logging.error(f"Error getting content similarity recommendations: {e}")
            return []

    def _blend_recommendations(self, cf_recs, seq_recs, content_recs, n=10, gcn_recs=None):
        """Blend CF, sequential, and content recommendations using Reciprocal Rank Fusion (RRF)."""
        k = 60
        rrf_scores = {}
        explanations = {}
        sources = {}
        
        def add_to_rrf(recs, weight=1.0):
            for rank, rec in enumerate(recs):
                item_id = rec['item_id']
                score = weight / (k + rank)
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + score
                
                # Keep explanation from the source with highest confidence score
                current_weight = explanations.get(item_id, {}).get("weight", 0.0)
                if weight > current_weight:
                    explanations[item_id] = {
                        "explanation": rec.get("explanation", ""),
                        "weight": weight
                    }
                    sources[item_id] = rec.get("source", "")
                
        if seq_recs:
            add_to_rrf(seq_recs, weight=1.5)
        if gcn_recs:
            add_to_rrf(gcn_recs, weight=1.2)
        if cf_recs:
            add_to_rrf(cf_recs, weight=1.0)
        if content_recs:
            add_to_rrf(content_recs, weight=0.8)
            
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_items:
            return []
            
        max_score = sorted_items[0][1]
        blended = []
        for item_id, score in sorted_items[:n]:
            blended.append({
                "item_id": item_id,
                "score": round(score / max_score, 4),
                "source": sources.get(item_id, ""),
                "explanation": explanations.get(item_id, {}).get("explanation", "Recommended for you.")
            })
        return blended

    def recommend(self, user_id, n=10, features=None, diversity=0.8, exclude_categories=None, exclude_items=None,
                  context_device=None, context_location=None, context_time=None,
                  w_relevance=None, w_freshness=None, w_fatigue=None, w_context=None, query=None, w_ssl=None):
        """Get blended or sequential recommendations according to A/B test cohort routing, with support for filtering, context, multi-objective linear fusion, and query steering."""
        start_time = time.time()
        import json
        
        # 1. Construct context vector for Neural Linear Bandit
        context_vector = self._get_context_vector(user_id, context_device, context_time)
        
        # 2. Select weights via Bandit or use custom overrides
        is_custom_weights = any(w is not None for w in [w_relevance, w_freshness, w_fatigue, w_context, w_ssl])
        if not is_custom_weights:
            arm_id, (w_rel, w_fresh, w_fatigue, w_context, w_rl, w_ssl_val) = self.bandit.select_action(context_vector)
            w_ssl = w_ssl_val
        else:
            arm_id = -1
            w_rel = w_relevance if w_relevance is not None else getattr(config, 'WEIGHT_RELEVANCE', 1.0)
            w_fresh = w_freshness if w_freshness is not None else getattr(config, 'WEIGHT_FRESHNESS', 0.2)
            w_fatigue = w_fatigue if w_fatigue is not None else getattr(config, 'WEIGHT_FATIGUE', 0.3)
            w_context = w_context if w_context is not None else getattr(config, 'WEIGHT_CONTEXT', 0.4)
            w_rl = 0.1  # default RL weight fallback
            w_ssl = w_ssl if w_ssl is not None else 0.15 # default SSL weight fallback

        with self._lock:
            # 3. Deterministic cohort routing
            cohort = ABTestManager.get_cohort(user_id)
            self.metrics.record_recommendation_served(cohort)
            
            cand_limit = max(n * 3, 50)

            # Get user's recent feedback
            user_feedback = []
            try:
                from data.feedback import get_user_feedback
                user_feedback = get_user_feedback(user_id, limit=50)
            except:
                pass

            # Parse query negations
            if query:
                import re
                q_lower = query.lower()
                neg_cats = re.findall(r"(?:no|not|without)\s+(movies|music|books|articles|news)", q_lower)
                if neg_cats:
                    if exclude_categories is None:
                        exclude_categories = set()
                    elif isinstance(exclude_categories, str):
                        exclude_categories = {c.strip().lower() for c in exclude_categories.split(",") if c.strip()}
                    else:
                        exclude_categories = set(exclude_categories)
                    exclude_categories.update(neg_cats)

            # Retrieve content similarity recommendations (supporting query steering)
            content_recs = []
            if query and FAISS_AVAILABLE and self.faiss_index is not None:
                try:
                    e_query = self.faiss_index._embedder.encode(query)
                    
                    e_history = np.zeros(384, dtype=np.float32)
                    history_count = 0
                    if user_feedback:
                        hist_iids = [f["item_id"] for f in user_feedback]
                        hist_details = get_items_by_ids(hist_iids)
                        for f in user_feedback:
                            item = hist_details.get(f["item_id"])
                            if item:
                                e_history += self.faiss_index._embedder.embed_item(item)
                                history_count += 1
                                
                    if history_count > 0:
                        e_history = e_history / history_count
                        norm = np.linalg.norm(e_history)
                        if norm > 0:
                            e_history = e_history / norm
                    else:
                        e_history = e_query.copy()
                        
                    alpha = 0.5
                    e_final = alpha * e_history + (1 - alpha) * e_query
                    norm = np.linalg.norm(e_final)
                    if norm > 0:
                        e_final = e_final / norm
                        
                    raw_recs = self.faiss_index.search(e_final, n=cand_limit)
                    for r in raw_recs:
                        content_recs.append({
                            "item_id": r["item_id"],
                            "score": r["score"],
                            "source": "intent_search",
                            "explanation": f"Matches your search query: '{query}'."
                        })
                except Exception as e:
                    logging.error(f"Error executing intent search query: {e}")
                    content_recs = self.get_content_similarity_recs(user_id, n=cand_limit)
            else:
                content_recs = self.get_content_similarity_recs(user_id, n=cand_limit)
            
            if cohort == "B":
                # Cohort B (Treatment): Sequential-Heavy. Try SASRec first (with dwell-time blended inside), fallback to FAISS similarity.
                seq_recs = self.get_sequential_recs(user_id, n=cand_limit)
                if seq_recs:
                    candidates = seq_recs
                else:
                    candidates = content_recs
            else:
                # Cohort A (Control): Standard RRF blending (GCN CF + CF + Sequential + Content)
                seq_recs = self.get_sequential_recs(user_id, n=cand_limit)
                gcn_recs = self.get_lightgcn_recs(user_id, n=cand_limit)
                cf_recs = []
                has_enough_feedback = len(user_feedback) >= self.cold_start_threshold

                if LIGHTFM_AVAILABLE and self.model is not None and user_id in self.user_id_map and has_enough_feedback:
                    try:
                        user_idx = self.user_id_map[user_id]
                        n_items = len(self.item_id_map)
                        scores = self.model.predict(
                            user_idx,
                            np.arange(n_items),
                            item_features=self.item_features
                        )
                        top_items_idx = np.argsort(-scores)[:cand_limit]
                        inv_item_map = {v: k for k, v in self.item_id_map.items()}
                        
                        # Tag profiles for explanations
                        user_tags = set()
                        for f in user_feedback:
                            item_details_f = get_items_by_ids([f['item_id']]).get(f['item_id'])
                            if item_details_f and item_details_f.get('tags'):
                                user_tags.update([t.strip() for t in item_details_f['tags'].split(',')])

                        for idx in top_items_idx:
                            item_id = inv_item_map[idx]
                            item_details_f = get_items_by_ids([item_id]).get(item_id)
                            
                            explanation = "Recommended based on similar users' preferences."
                            if item_details_f and item_details_f.get('tags') and user_tags:
                                overlap = {t.strip() for t in item_details_f['tags'].split(',')} & user_tags
                                if overlap:
                                    explanation = f"Recommended based on your interest in tags: {', '.join(list(overlap)[:3])}."

                            cf_recs.append({
                                "item_id": item_id,
                                "score": float(scores[idx]),
                                "source": "collaborative",
                                "explanation": explanation
                            })
                    except Exception as e:
                        logging.error(f"Error predicting with LightFM: {e}")

                candidates = self._blend_recommendations(cf_recs, seq_recs, content_recs, n=cand_limit, gcn_recs=gcn_recs)
            
            # Cold-start fallback (e.g. user has absolutely no history)
            if not candidates:
                candidates = self.cold_start_recommend(tags_or_category=features, n=cand_limit)

            # Normalize exclusions
            ex_items = set()
            if exclude_items:
                if isinstance(exclude_items, str):
                    ex_items = {i.strip() for i in exclude_items.split(",") if i.strip()}
                else:
                    ex_items = set(exclude_items)

            ex_categories = set()
            if exclude_categories:
                if isinstance(exclude_categories, str):
                    ex_categories = {c.strip().lower() for c in exclude_categories.split(",") if c.strip()}
                else:
                    ex_categories = {c.lower() for c in exclude_categories}

            # Filter candidates based on exclusions and fetch item details unconditionally for re-ranking
            if candidates:
                item_ids = [c["item_id"] for c in candidates]
                item_details = get_items_by_ids(item_ids)
            else:
                item_details = {}

            filtered_candidates = []
            for c in candidates:
                item_id = c["item_id"]
                if item_id in ex_items:
                    continue
                if ex_categories:
                    details = item_details.get(item_id)
                    if details:
                        cat = (details.get("category") or "").lower()
                        if cat in ex_categories:
                            continue
                filtered_candidates.append(c)
            candidates = filtered_candidates

            user_interacted_items = [f["item_id"] for f in user_feedback]
            user_item_details = get_items_by_ids(user_interacted_items) if user_interacted_items else {}
            
            category_counts = {}
            for f in user_feedback:
                iid = f["item_id"]
                details = user_item_details.get(iid)
                if details:
                    cat = (details.get("category") or "").lower()
                    if cat:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
            max_count = max(category_counts.values()) if category_counts else 1

            # Prepare BCQ state vector
            state_vec = self._get_bcq_state(user_id, user_feedback)

            # Compute BEST-Rec universal user session embedding
            e_u = np.zeros(256, dtype=np.float32)
            if self.best_rec_model is not None and user_feedback:
                try:
                    history_iids = [f["item_id"] for f in user_feedback[-30:]]
                    history_indices = [self.seq_item_to_idx.get(iid, 0) for iid in history_iids if iid in self.seq_item_to_idx]
                    pad_len = max(0, 30 - len(history_indices))
                    padded = [0] * pad_len + history_indices[-30:]
                    
                    input_tensor = torch.tensor([padded], dtype=torch.long)
                    self.best_rec_model.eval()
                    with torch.no_grad():
                        u_emb, _, _ = self.best_rec_model(input_tensor)
                        e_u = u_emb.squeeze(0).numpy()
                except Exception as e:
                    logging.error(f"Error computing BEST-Rec user embedding: {e}")

            # Multi-Objective Linear Fusion Re-ranking
            re_ranked_candidates = []
            for c in candidates:
                item_id = c["item_id"]
                item = item_details.get(item_id) or {}
                
                relevance = c.get("score") or 0.0
                freshness = get_freshness_score(item)
                
                cat = (item.get("category") or "").lower()
                fatigue = (category_counts.get(cat, 0) / max_count) if (category_counts and max_count > 0) else 0.0
                
                context_match = get_context_match_score(item, context_device, context_location, context_time)
                
                # Compute BCQ RL Long-Term Q-value
                q_val = 0.0
                if self.bcq_model is not None and self.faiss_index is not None:
                    try:
                        item_emb = self.faiss_index._embedder.embed_item(item)
                        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                        item_tensor = torch.tensor(item_emb, dtype=torch.float32).unsqueeze(0)
                        # Predict Q-value
                        with torch.no_grad():
                            q_val = float(self.bcq_model(state_tensor, item_tensor).item())
                    except:
                        pass

                # Compute BEST-Rec universal representation similarity
                ssl_score = 0.0
                if self.best_rec_model is not None and item_id in self.seq_item_to_idx:
                    try:
                        item_idx = self.seq_item_to_idx[item_id]
                        self.best_rec_model.eval()
                        with torch.no_grad():
                            item_emb = self.best_rec_model.item_embedding(torch.tensor([item_idx], dtype=torch.long))
                            proj_emb = F.normalize(self.best_rec_model.ssl_projection(item_emb.squeeze(0)), p=2, dim=1)
                            ssl_score = float(torch.dot(torch.tensor(e_u), proj_emb[0]).item())
                    except Exception as e:
                        pass

                # Final linear fusion score including BCQ RL bonus and BEST-Rec similarity
                final_score = (relevance * w_rel) + (freshness * w_fresh) - (fatigue * w_fatigue) + (context_match * w_context) + (q_val * w_rl) + (ssl_score * w_ssl)
                
                c_copy = c.copy()
                c_copy["score"] = round(final_score, 4)
                c_copy["metrics"] = {
                    "relevance": float(relevance),
                    "freshness": float(freshness),
                    "fatigue": float(fatigue),
                    "context": float(context_match),
                    "ssl": float(ssl_score)
                }
                re_ranked_candidates.append(c_copy)
                
            # Apply Fairness Auditor PID adjustments on scored candidates
            self.fairness_auditor.audit_and_update_pid()
            for c in re_ranked_candidates:
                item = item_details.get(c["item_id"]) or {}
                cat = (item.get("category") or "").lower()
                c["score"] = round(self.fairness_auditor.get_fairness_score(cat, c["score"]), 4)

            # Apply Bayesian Surprise exploration decay / PID adjustment
            actual_kl = 0.0
            surprise_error = 0.0
            adjusted_diversity = diversity
            if hasattr(self, 'surprise_controller') and self.surprise_controller is not None and user_feedback:
                try:
                    adjusted_diversity, actual_kl, surprise_error = self.surprise_controller.adjust_diversity_lambda(
                        diversity, user_feedback, re_ranked_candidates, item_details
                    )
                except Exception as e:
                    logging.error(f"Error in Bayesian surprise PID controller: {e}")

            # Sort by new multi-objective score
            candidates = sorted(re_ranked_candidates, key=lambda x: x["score"], reverse=True)
                
            # MMR diversity re-ranking
            if adjusted_diversity < 1.0 and candidates:
                final_recs = mmr_rerank(candidates, item_details, n=n, diversity_lambda=adjusted_diversity)
            else:
                final_recs = candidates[:n]

            # Generate baseline candidate metrics for exact Shapley calculations
            baseline_metrics = {
                'relevance': 0.0,
                'freshness': 0.0,
                'fatigue': 0.0,
                'context': 0.0,
                'ssl': 0.0
            }
            if candidates:
                for c in candidates:
                    for k in baseline_metrics:
                        baseline_metrics[k] += c["metrics"][k]
                for k in baseline_metrics:
                    baseline_metrics[k] /= len(candidates)

            # Counterfactual Explainability (LIME for Cohort B / Shapley for Cohort A)
            weights = {
                'w_relevance': w_rel,
                'w_freshness': w_fresh,
                'w_fatigue': w_fatigue,
                'w_context': w_context,
                'w_ssl': w_ssl
            }

            for idx_rec, r in enumerate(final_recs):
                # Cohort B: sequence explainability using LIME (top 5 only)
                if cohort == "B" and self.seq_model is not None and idx_rec < 5:
                    try:
                        from sessions.session_builder import get_user_sequence
                        user_history = get_user_sequence(user_id, max_len=50)
                        user_history_indices = [self.seq_item_to_idx.get(iid, 0) for iid in user_history if iid in self.seq_item_to_idx]
                        user_history_indices = [idx for idx in user_history_indices if idx != 0]
                        target_item_idx = self.seq_item_to_idx.get(r["item_id"], 0)
                        
                        explanation = SASRecLimeExplainer.explain_sequence(
                            model=self.seq_model,
                            user_history_indices=user_history_indices,
                            idx_to_item=self.seq_idx_to_item,
                            item_details=item_details,
                            target_item_idx=target_item_idx,
                            num_perturbations=15,
                            drop_prob=0.2
                        )
                        r["explanation"] = explanation
                    except Exception as e:
                        logging.error(f"Error computing sequence LIME explanation: {e}")
                else:
                    # Cohort A / default: Exact Linear Shapley Value attribution
                    try:
                        _, _, explanation = LinearShapleyExplainer.explain_recommendation(
                            r["item_id"], r["metrics"], baseline_metrics, weights,
                            context_device=context_device, context_location=context_location, context_time=context_time
                        )
                        r["explanation"] = explanation
                    except Exception as e:
                        logging.error(f"Error computing Shapley explanation: {e}")
                
                # Delete auxiliary metrics dict to keep output clean
                r.pop("metrics", None)

            # Record recommended items details for rolling DI calculation
            recommended_details = [item_details.get(r["item_id"]) for r in final_recs if item_details.get(r["item_id"])]
            self.fairness_auditor.record_recommendations(recommended_details)

            # Record served recommendations for CTR tracking
            self.metrics.record_served_recs(user_id, [r["item_id"] for r in final_recs], cohort, engine=self)

            # Track active bandit arms for served items
            if arm_id != -1:
                now = time.time()
                for r in final_recs:
                    self.metrics.served_bandit_arms[(user_id, r["item_id"])] = (arm_id, context_vector, now)

            # Log impressions for causal propensity model estimation
            for r in final_recs:
                try:
                    context_data = {"device": context_device, "time_of_day": context_time, "location": context_location}
                    add_impression(user_id, r["item_id"], cohort, json.dumps(context_data))
                except:
                    pass

            # Record latency
            self.metrics.record_latency("recommend", time.time() - start_time)
            
            return final_recs


    def similar_items(self, item_id, n=5, exclude_categories=None, exclude_items=None):
        """Find items similar to a given item, with exclusion filters."""
        # Normalize exclusions
        ex_items = set()
        if exclude_items:
            if isinstance(exclude_items, str):
                ex_items = {i.strip() for i in exclude_items.split(",") if i.strip()}
            else:
                ex_items = set(exclude_items)
        ex_items.add(item_id)

        ex_categories = set()
        if exclude_categories:
            if isinstance(exclude_categories, str):
                ex_categories = {c.strip().lower() for c in exclude_categories.split(",") if c.strip()}
            else:
                ex_categories = {c.lower() for c in exclude_categories}

        with self._lock:
            candidates = []
            
            # Try FAISS content similarity first
            if FAISS_AVAILABLE and self.faiss_index is not None:
                try:
                    from data.items import get_item
                    item_details = get_item(item_id)
                    if item_details:
                        embedding = self.faiss_index._embedder.embed_item(item_details)
                        raw_recs = self.faiss_index.search(embedding, n=n + 50)
                        for r in raw_recs:
                            if r['item_id'] != item_id:
                                candidates.append({
                                    "item_id": r["item_id"],
                                    "score": r["score"],
                                    "source": "content_similarity",
                                    "explanation": f"Similar to '{item_details.get('title', 'queried item')}'."
                                })
                except Exception as e:
                    logging.error(f"Error getting similar items from FAISS: {e}")

            # Fallback to LightFM representation similarity if FAISS failed/not available
            if not candidates and LIGHTFM_AVAILABLE and item_id in self.item_id_map:
                try:
                    item_idx = self.item_id_map[item_id]
                    _, item_embeddings = self.model.get_item_representations(self.item_features)
                    target_embedding = item_embeddings[item_idx]

                    scores = np.dot(item_embeddings, target_embedding) / (
                        np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                    )

                    top_items_idx = np.argsort(-scores)[1:n+50]  # Skip itself
                    inv_item_map = {v: k for k, v in self.item_id_map.items()}

                    for idx in top_items_idx:
                        candidates.append({
                            "item_id": inv_item_map[idx],
                            "score": float(scores[idx]),
                            "source": "collaborative_similarity",
                            "explanation": f"Often liked by users who liked '{item_id}'."
                        })
                except Exception as e:
                    logging.error(f"Error getting similar items from LightFM: {e}")

            if not candidates:
                return []

            # Filter candidates based on exclusions
            item_ids = [c["item_id"] for c in candidates]
            item_details = get_items_by_ids(item_ids)

            filtered_recs = []
            for c in candidates:
                iid = c["item_id"]
                if iid in ex_items:
                    continue
                if ex_categories:
                    details = item_details.get(iid)
                    if details:
                        cat = (details.get("category") or "").lower()
                        if cat in ex_categories:
                            continue
                filtered_recs.append(c)
                if len(filtered_recs) >= n:
                    break
            
            return filtered_recs

    def cold_start_recommend(self, tags_or_category=None, n=10):
        """Get recommendations for a cold-start user using content metadata similarity (FAISS) or popularity."""
        if tags_or_category and FAISS_AVAILABLE and self.faiss_index is not None:
            try:
                query = tags_or_category if isinstance(tags_or_category, str) else " ".join(tags_or_category)
                recs = self.faiss_index.search_by_text(query, n=n)
                if recs:
                    return [{
                        "item_id": r["item_id"],
                        "score": r["score"],
                        "source": "cold_start_content",
                        "explanation": f"Matches your interest in '{query}'."
                    } for r in recs]
            except Exception as e:
                logging.error(f"Error in FAISS cold start recommend: {e}")

        items = []
        if tags_or_category:
            if isinstance(tags_or_category, str):
                tags = [t.strip() for t in tags_or_category.split(',')]
            else:
                tags = tags_or_category
                
            for tag in tags:
                items.extend(search_by_tags(tag))
        
        if not items:
            feedback = get_all_feedback()
            item_counts = {}
            for f in feedback:
                item_counts[f['item_id']] = item_counts.get(f['item_id'], 0) + 1
            
            sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
            return [{
                "item_id": i[0],
                "score": float(i[1]),
                "source": "popularity",
                "explanation": "Trending among all users."
            } for i in sorted_items[:n]]
            
        seen = set()
        unique_items = []
        for item in items:
            if item['item_id'] not in seen:
                unique_items.append({
                    "item_id": item['item_id'],
                    "score": 1.0,
                    "source": "cold_start_tag",
                    "explanation": f"Matches your interest in tag '{item.get('tags', '')}'."
                })
                seen.add(item['item_id'])
                
        return unique_items[:n]
