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

    def predict(self, data: Data, target_idx: int) -> dict:
        """
        Run RGCN inference on a mini-graph.
        
        Args:
            data: PyG Data object with x, edge_index, edge_type
            target_idx: index of the target node in the graph
        
        Returns:
            dict with label, prob_human, prob_bot
        """
        data = data.to(self.device)

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_type)
            target_logits = logits[target_idx]
            probs = F.softmax(target_logits, dim=0)

            pred_class = probs.argmax().item()
            prob_human = round(probs[0].item(), 4)
            prob_bot = round(probs[1].item(), 4)

        label = "bot" if pred_class == 1 else "human"

        logger.info(
            f"Prediction: {label} (human={prob_human}, bot={prob_bot})"
        )

        return {
            "label_pred": label,
            "prob_human": prob_human,
            "prob_bot": prob_bot,
            "confidence": round(max(prob_human, prob_bot), 4),
        }

    def predict_from_request(self, request_data: dict) -> dict:
        """
        Full pipeline: request JSON → graph → RGCN → prediction.
        
        Args:
            request_data: dict with target, neighbors, relations
        
        Returns:
            dict with prediction results
        """
        data, target_idx = build_mini_graph(request_data)
        result = self.predict(data, target_idx)

        # Add metadata
        result["graph_info"] = {
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
        }

        return result
