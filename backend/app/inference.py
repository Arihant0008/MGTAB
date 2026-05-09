"""
Inference engine — loads the trained RGCN model once and
provides a predict() method for incoming requests.
"""

import torch
import torch.nn.functional as F
import logging
from torch_geometric.data import Data

from .config import MODEL_PATH, NUM_FEATURES, HIDDEN_DIM, NUM_CLASSES, NUM_RELATIONS
from .rgcn_model import RGCN
from .graph_builder import build_mini_graph

logger = logging.getLogger(__name__)

# ── Confidence calibration bounds ────────────────────────────────────────────
# Prevents the model from outputting 0% or 100% when the input graph is
# degenerate (too few nodes/edges). The trained RGCN was never exposed to
# 1-node ego-graphs during training, so extreme outputs on sparse graphs are
# artefacts of distribution shift, not genuine high confidence.
PROB_CLIP_MIN = 0.05   # never report less than 5% for either class
PROB_CLIP_MAX = 0.95   # never report more than 95% for either class

# Graph quality thresholds
MIN_NODES_FOR_HIGH_CONFIDENCE = 5   # fewer → warn
MIN_EDGES_FOR_HIGH_CONFIDENCE = 5   # fewer → warn

# ── High-follower calibration ─────────────────────────────────────────────────
# The MGTAB training set contained mostly regular users and social-media bots.
# Celebrity / politician / brand accounts (>1M followers) are genuinely
# out-of-distribution: their follower/friend ratio can exceed 10,000:1 which
# the model has never seen in training. This causes extreme bot scores for
# accounts like @narendramodi (106M followers) or @ViratKohli.
#
# Heuristic: accounts with verified follower counts above HIGH_FOLLOWER_THRESHOLD
# are blended toward a human prior. The blend weight scales with follower count
# so 1M gets a light correction and 100M gets a strong one.
HIGH_FOLLOWER_THRESHOLD = 1_000_000    # 1M followers → start blending
HIGH_FOLLOWER_MAX       = 50_000_000   # 50M+ followers → max blend weight

# At threshold: blend_weight = 0.30  (30% human prior)
# At maximum:   blend_weight = 0.85  (85% human prior) — accounts with 50M+ followers
#               are overwhelmingly real: politicians, celebrities, global brands.
HIGH_FOLLOWER_BLEND_MIN = 0.30
HIGH_FOLLOWER_BLEND_MAX = 0.85



class InferenceEngine:
    """
    Singleton-style inference engine.
    Loads best_rgcn.pt at initialization and provides prediction.
    """

    def __init__(self):
        self.device = torch.device("cpu")  # CPU inference for simplicity
        self.model = self._load_model()

    def _load_model(self) -> RGCN:
        """Load the trained RGCN model from checkpoint."""
        logger.info(f"Loading RGCN model from {MODEL_PATH}...")

        model = RGCN(
            num_features=NUM_FEATURES,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            num_relations=NUM_RELATIONS,
        )

        state_dict = torch.load(
            str(MODEL_PATH),
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        logger.info("RGCN model loaded successfully.")
        return model

    def predict(self, data: Data, target_idx: int, followers_count: int = 0) -> dict:
        """
        Run RGCN inference on a mini-graph.
        
        Args:
            data:            PyG Data object with x, edge_index, edge_type
            target_idx:      index of the target node in the graph
            followers_count: raw follower count of the target user (used for
                             high-follower calibration — celebrity OOD correction)
        
        Returns:
            dict with label, prob_human, prob_bot, confidence, quality_warning
        """
        data = data.to(self.device)

        # ── Detect degenerate graph BEFORE inference ──────────────────────
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        is_degenerate = (
            num_nodes < MIN_NODES_FOR_HIGH_CONFIDENCE
            or num_edges < MIN_EDGES_FOR_HIGH_CONFIDENCE
        )

        if is_degenerate:
            logger.warning(
                f"Degenerate graph detected: {num_nodes} nodes, {num_edges} edges. "
                f"Prediction reliability is reduced. "
                f"This typically means Scweet could not scrape enough neighbors "
                f"(rate-limited or API failure)."
            )

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_type)
            target_logits = logits[target_idx]
            probs = F.softmax(target_logits, dim=0)

            raw_prob_human = probs[0].item()
            raw_prob_bot   = probs[1].item()

        # ── Standard confidence calibration ──────────────────────────────
        # Clamp to [PROB_CLIP_MIN, PROB_CLIP_MAX] to prevent overconfident
        # predictions on degenerate (sparse) graphs.
        prob_human = max(PROB_CLIP_MIN, min(PROB_CLIP_MAX, raw_prob_human))
        prob_bot   = max(PROB_CLIP_MIN, min(PROB_CLIP_MAX, raw_prob_bot))

        # ── High-follower calibration (celebrity OOD correction) ──────────
        # The MGTAB training set never saw accounts with >1M followers.
        # At that scale, follower/friend ratios (e.g. 39,000:1 for Modi)
        # look bot-like to the RGCN but are actually a hallmark of real
        # public figures. We blend toward a human prior proportional to
        # how far above the threshold the follower count is.
        high_follower_warning = None
        if followers_count >= HIGH_FOLLOWER_THRESHOLD:
            # Linear interpolation of blend weight
            # followers at threshold → blend_min, at max → blend_max
            t = min(
                1.0,
                (followers_count - HIGH_FOLLOWER_THRESHOLD)
                / (HIGH_FOLLOWER_MAX - HIGH_FOLLOWER_THRESHOLD),
            )
            blend_weight = HIGH_FOLLOWER_BLEND_MIN + t * (
                HIGH_FOLLOWER_BLEND_MAX - HIGH_FOLLOWER_BLEND_MIN
            )

            # Human prior: 0.90 human / 0.10 bot
            # Accounts with 1M+ real followers are ~90% guaranteed to be real people.
            # (State-sponsored mega-bots with 1M+ real followers are exceedingly rare.)
            prior_human = 0.90
            prior_bot   = 0.10

            blended_human = (1 - blend_weight) * prob_human + blend_weight * prior_human
            blended_bot   = (1 - blend_weight) * prob_bot   + blend_weight * prior_bot

            followers_m = followers_count / 1_000_000
            logger.info(
                f"High-follower calibration applied: "
                f"{followers_m:.1f}M followers, blend_weight={blend_weight:.2f}. "
                f"Bot prob: {prob_bot:.4f} -> {blended_bot:.4f}"
            )

            prob_human = blended_human
            prob_bot   = blended_bot

            high_follower_warning = (
                f"This account has {followers_m:.1f}M followers. "
                "High-follower public figures (celebrities, politicians, brands) "
                "have extreme follower ratios that fall outside the model's training "
                "distribution. Results are adjusted with a human-prior correction."
            )

        # Re-normalise so they still sum to 1.0
        total = prob_human + prob_bot
        prob_human = round(prob_human / total, 4)
        prob_bot   = round(prob_bot   / total, 4)

        pred_class = 1 if prob_bot > prob_human else 0
        label = "bot" if pred_class == 1 else "human"

        logger.info(
            f"Final prediction: {label} "
            f"(raw: human={raw_prob_human:.4f}, bot={raw_prob_bot:.4f}) "
            f"(final: human={prob_human}, bot={prob_bot}) "
            f"nodes={num_nodes}, edges={num_edges}, followers={followers_count:,}"
        )

        # ── Build quality warning ─────────────────────────────────────────
        quality_warning = None
        if high_follower_warning:
            quality_warning = high_follower_warning
        elif is_degenerate:
            quality_warning = (
                f"Low graph coverage ({num_nodes} nodes, {num_edges} edges). "
                "Confidence may be reduced. Try again if scraping was rate-limited."
            )

        return {
            "label_pred": label,
            "prob_human": prob_human,
            "prob_bot":   prob_bot,
            "confidence": round(max(prob_human, prob_bot), 4),
            "quality_warning": quality_warning,
        }

    def predict_from_request(self, request_data: dict) -> dict:
        """
        Full pipeline: request JSON -> graph -> RGCN -> prediction.

        Extracts followers_count from the target profile so the
        high-follower calibration heuristic can be applied.
        """
        # Extract followers_count for calibration BEFORE building the graph
        target_profile = request_data.get("target", {}).get("profile", {})
        followers_count = int(target_profile.get("followers_count") or 0)

        data, target_idx, graph_quality = build_mini_graph(request_data)
        result = self.predict(data, target_idx, followers_count=followers_count)

        # Add metadata including graph quality diagnostics
        result["graph_info"] = {
            "num_nodes":          data.num_nodes,
            "num_edges":          data.num_edges,
            "nodes_with_tweets":  graph_quality.get("nodes_with_tweets", 0),
            "nodes_profile_only": graph_quality.get("nodes_profile_only", 0),
            "nodes_no_data":      graph_quality.get("nodes_no_data", 0),
            "edges_filtered":     graph_quality.get("edges_filtered", 0),
        }

        # Merge graph-builder quality warning only if no higher-priority
        # warning (e.g. high-follower) was already set
        if not result.get("quality_warning") and graph_quality.get("warning"):
            result["quality_warning"] = graph_quality["warning"]

        return result