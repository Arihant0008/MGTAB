"""
Graph builder — constructs a PyTorch Geometric Data object
from the user's request JSON.
Handles:
 - Target node (always present)
 - Neighbor nodes (optional, zero-vector placeholder if no data provided)
 - Relations/edges mapped to MGTAB relation IDs 0-6
Key design decisions (per MGTAB paper):
 - Edges to neighbors WITHOUT real profile data are SKIPPED. The trained
   RGCN learned with all nodes having real 788-dim features; injecting
   zero-vector neighbors corrupts the mean aggregation and produces
   random-seeming outputs.
 - Edge directions follow Paper Table 4:
     follower: neighbor → target  (neighbor follows the target)
     friend:   target → neighbor  (target follows the neighbor)
     mention:  target → neighbor
     reply:    target → neighbor
     quoted:   target → neighbor
 - URL and hashtag are UNDIRECTED (bidirectional edges).
"""

import torch
import numpy as np
import logging
from torch_geometric.data import Data

from .config import RELATION_MAP, NUM_FEATURES, REVERSE_SOURCE_RELATIONS, UNDIRECTED_RELATIONS
from .features import build_node_feature

logger = logging.getLogger(__name__)


def _has_real_data(neighbor: dict) -> bool:
    """Check if a neighbor has actual profile or tweet data (not just an ID)."""
    profile = neighbor.get("profile")
    tweets = neighbor.get("tweets", [])

    if tweets and any(t.strip() for t in tweets if isinstance(t, str)):
        return True

    if profile and isinstance(profile, dict):
        # Check if any meaningful profile field is set
        meaningful_fields = [
            "followers_count", "friends_count", "listed_count",
            "statuses_count", "favourites_count", "name", "screen_name",
            "description", "created_at",
        ]
        for field in meaningful_fields:
            val = profile.get(field)
            if val and val != 0 and val != "" and val is not None:
                return True

    return False


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
    # ── 1. Collect neighbor data ─────────────────────────────────────
    neighbors = request_data.get("neighbors", [])
    relations = request_data.get("relations", [])

    # Build a map of neighbor ID → neighbor data (only those with real data)
    neighbor_data_map = {}
    for neighbor in neighbors:
        nid = neighbor.get("id", "")
        if nid and _has_real_data(neighbor):
            neighbor_data_map[nid] = neighbor

    # ── 2. Filter relations: only keep edges where neighbor has real data ─
    valid_relations = []
    for rel in relations:
        rel_type = rel.get("relation", "").lower().strip()
        if rel_type not in RELATION_MAP:
            logger.warning(f"Unknown relation type '{rel_type}', skipping.")
            continue

        src_raw = rel.get("source", "")
        dst_raw = rel.get("target", "")

        # Skip relations with empty target/source
        if not src_raw or not dst_raw:
            continue

        # Identify which end is the target user and which is the neighbor
        target_aliases = {"target", "__target__", "this_user", "self"}
        src_is_target = src_raw.lower() in target_aliases
        dst_is_target = dst_raw.lower() in target_aliases

        if src_is_target:
            neighbor_id = dst_raw
        elif dst_is_target:
            neighbor_id = src_raw
        else:
            # Neither end is the target — skip in single-user inference
            logger.warning(f"Skipping relation with no target user: {src_raw} → {dst_raw}")
            continue

        # Only keep edges where the neighbor has real feature data
        if neighbor_id not in neighbor_data_map:
            logger.info(
                f"Skipping '{rel_type}' edge to '{neighbor_id}': "
                f"no profile/tweet data provided (zero-vector neighbors corrupt RGCN predictions)."
            )
            continue

        valid_relations.append({
            "neighbor_id": neighbor_id,
            "relation": rel_type,
            "src_is_target": src_is_target,
        })

    # ── 3. Build node index mapping ──────────────────────────────────
    node_ids = ["__target__"]
    neighbor_ids_used = set()

    for vr in valid_relations:
        nid = vr["neighbor_id"]
        if nid not in neighbor_ids_used:
            neighbor_ids_used.add(nid)
            node_ids.append(nid)

    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    logger.info(
        f"Building mini-graph: {num_nodes} nodes, "
        f"{len(valid_relations)} valid relations "
        f"(filtered from {len(relations)} total)."
    )

    # ── 4. Build node features ───────────────────────────────────────
    features_list = []

    # Target node (index 0)
    target_data = request_data.get("target", {})
    target_profile = target_data.get("profile", {})
    target_tweets = target_data.get("tweets", [])
    target_feature = build_node_feature(target_profile, target_tweets)
    features_list.append(target_feature)

    # Neighbor nodes (only those with real data)
    for nid in node_ids[1:]:
        n = neighbor_data_map[nid]  # guaranteed to exist from filtering above
        n_profile = n.get("profile", {})
        n_tweets = n.get("tweets", [])
        feat = build_node_feature(n_profile, n_tweets)
        features_list.append(feat)

    x = torch.tensor(np.stack(features_list), dtype=torch.float32)

    # ── 5. Build edges with correct directions ───────────────────────
    src_list = []
    dst_list = []
    etype_list = []

    target_idx = 0

    for vr in valid_relations:
        neighbor_idx = node_id_to_idx[vr["neighbor_id"]]
        rel_type = vr["relation"]
        etype_idx = RELATION_MAP[rel_type]

        # Determine edge direction per Paper Table 4
        if rel_type in REVERSE_SOURCE_RELATIONS:
            # "follower": target is followed BY neighbor → edge: neighbor → target
            src_idx = neighbor_idx
            dst_idx = target_idx
        else:
            # "friend", "mention", "reply", "quoted": target acts on neighbor → edge: target → neighbor
            src_idx = target_idx
            dst_idx = neighbor_idx

        src_list.append(src_idx)
        dst_list.append(dst_idx)
        etype_list.append(etype_idx)

        # For undirected relations (URL, hashtag), add reverse edge
        if rel_type in UNDIRECTED_RELATIONS:
            src_list.append(dst_idx)
            dst_list.append(src_idx)
            etype_list.append(etype_idx)

    if not src_list:
        # No valid edges — create a self-loop so RGCN can still run.
        # The model's root_weight handles self-connection; the self-loop
        # with type 0 adds a small additional self-aggregation term.
        logger.info(
            "No valid edges (no neighbors with real data). "
            "Using self-loop → model operates in feature-only mode."
        )
        src_list = [0]
        dst_list = [0]
        etype_list = [0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(etype_list, dtype=torch.long)

    # ── 6. Assemble PyG Data ─────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    logger.info(
        f"Mini-graph built: {data.num_nodes} nodes, {data.num_edges} edges "
        f"(types: {sorted(set(etype_list))})"
    )
    return data, target_idx
