"""
Graph builder — constructs a PyTorch Geometric Data object
from the user's request JSON.

Handles:
 - Target node (always present)
 - Neighbor nodes (optional, zero-vector placeholder if no data provided)
 - Relations/edges mapped to MGTAB relation IDs 0-6
"""

import torch
import numpy as np
import logging
from torch_geometric.data import Data

from .config import RELATION_MAP, NUM_FEATURES
from .features import build_node_feature

logger = logging.getLogger(__name__)


def build_mini_graph(request_data: dict) -> tuple[Data, int]:
    """
    Build a small PyG graph from the prediction request.
    
    Args:
        request_data: dict with keys:
            - target: {profile: {...}, tweets: [...]}
            - neighbors: [{id: str, profile: {...}, tweets: [...]}]  (optional)
            - relations: [{source: str, target: str, relation: str}] (optional)
    
    Returns:
        (data, target_idx): PyG Data object and the index of the target node.
    """
    # ── 1. Build node index mapping ──────────────────────────────────
    node_ids = ["__target__"]
    neighbors = request_data.get("neighbors", [])
    relations = request_data.get("relations", [])

    # Collect unique neighbor IDs
    neighbor_id_set = set()
    for neighbor in neighbors:
        nid = neighbor.get("id", "")
        if nid and nid not in neighbor_id_set:
            neighbor_id_set.add(nid)
            node_ids.append(nid)

    # Also add IDs from relations that aren't yet in node_ids
    for rel in relations:
        for key in ["source", "target"]:
            rid = rel.get(key, "")
            # Map "target" keyword to __target__
            if rid in ("target", "__target__", "this_user", "self"):
                continue
            if rid and rid not in neighbor_id_set:
                neighbor_id_set.add(rid)
                node_ids.append(rid)

    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    logger.info(f"Building mini-graph with {num_nodes} nodes and {len(relations)} relation entries.")

    # ── 2. Build node features ───────────────────────────────────────
    features_list = []

    # Target node (index 0)
    target_data = request_data.get("target", {})
    target_profile = target_data.get("profile", {})
    target_tweets = target_data.get("tweets", [])
    target_feature = build_node_feature(target_profile, target_tweets)
    features_list.append(target_feature)

    # Neighbor nodes
    neighbor_data_map = {}
    for neighbor in neighbors:
        nid = neighbor.get("id", "")
        if nid:
            neighbor_data_map[nid] = neighbor

    for nid in node_ids[1:]:  # skip __target__
        if nid in neighbor_data_map:
            n = neighbor_data_map[nid]
            n_profile = n.get("profile", {})
            n_tweets = n.get("tweets", [])
            feat = build_node_feature(n_profile, n_tweets)
        else:
            # No data for this neighbor → zero vector placeholder
            feat = np.zeros(NUM_FEATURES, dtype=np.float32)
        features_list.append(feat)

    x = torch.tensor(np.stack(features_list), dtype=torch.float32)

    # ── 3. Build edges ───────────────────────────────────────────────
    src_list = []
    dst_list = []
    etype_list = []

    def resolve_id(raw_id: str) -> str:
        """Map target-like IDs to __target__."""
        if raw_id in ("target", "__target__", "this_user", "self"):
            return "__target__"
        return raw_id

    for rel in relations:
        src_id = resolve_id(rel.get("source", ""))
        dst_id = resolve_id(rel.get("target", ""))
        rel_type = rel.get("relation", "").lower().strip()

        if src_id not in node_id_to_idx or dst_id not in node_id_to_idx:
            logger.warning(f"Skipping edge {src_id} → {dst_id}: node not found.")
            continue

        if rel_type not in RELATION_MAP:
            logger.warning(f"Unknown relation type '{rel_type}', skipping.")
            continue

        src_idx = node_id_to_idx[src_id]
        dst_idx = node_id_to_idx[dst_id]
        etype_idx = RELATION_MAP[rel_type]

        src_list.append(src_idx)
        dst_list.append(dst_idx)
        etype_list.append(etype_idx)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(etype_list, dtype=torch.long)
    else:
        # No edges — create a self-loop for the target node so RGCN can still run
        logger.warning("No valid edges provided. Adding self-loop for target node.")
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_type = torch.tensor([0], dtype=torch.long)

    # ── 4. Assemble PyG Data ─────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    target_idx = 0  # target is always the first node

    logger.info(f"Mini-graph built: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data, target_idx
