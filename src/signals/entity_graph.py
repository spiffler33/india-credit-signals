# WHY THIS: Entity contagion graph for propagating credit signals between NBFCs.
# v1 uses subsector-only edge weights (same subsector = 0.8, cross = 0.1).
# The graph loads from nbfc_entities.yaml (our 44-entity master list) and provides
# peer queries like "who is connected to DHFL?" for the propagation engine.
#
# Why not NetworkX? For v1, we only need adjacency lookup and peer sorting.
# A simple dict-of-dicts is faster, has zero dependencies, and is easier to test.
# If v2 needs centrality metrics (PageRank, betweenness), upgrade to NetworkX then.

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# ============================================================
# Data Structures
# ============================================================

@dataclass
class EntityNode:
    """A single NBFC entity in the contagion graph.

    Wraps the data from nbfc_entities.yaml with subsector assignment.
    """
    name: str
    full_name: str
    subsector: str
    status: str
    aliases: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"EntityNode({self.name!r}, subsector={self.subsector!r})"


@dataclass
class ContagionEdge:
    """A weighted edge between two entities in the contagion graph.

    The `components` dict stores the breakdown of the weight (v2-ready):
    v1: {"subsector": 0.8}
    v2: {"subsector": 0.8, "funding_similarity": 0.3} → total = weighted sum
    """
    source: str
    target: str
    weight: float
    components: dict[str, float] = field(default_factory=dict)


class EntityGraph:
    """Weighted undirected graph of NBFC entities for contagion propagation.

    # 🎓 WHY a graph? Credit crises propagate through sector connections.
    # When DHFL (housing finance) collapsed, other housing finance NBFCs
    # were hit hardest because they shared the same funding markets, asset
    # classes, and investor base. A graph captures these connections as
    # weighted edges — higher weight = stronger contagion channel.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, EntityNode] = {}
        self.edges: dict[str, dict[str, ContagionEdge]] = {}  # adjacency: {source: {target: edge}}
        self._alias_map: dict[str, str] = {}  # lowercase alias → canonical name

    @property
    def alias_map(self) -> dict[str, str]:
        return self._alias_map

    def add_node(self, node: EntityNode) -> None:
        """Add an entity node and register its aliases."""
        self.nodes[node.name] = node
        if node.name not in self.edges:
            self.edges[node.name] = {}

        # Register aliases for fuzzy matching
        for alias in node.aliases:
            self._alias_map[alias.lower()] = node.name
        # Also register the canonical name itself
        self._alias_map[node.name.lower()] = node.name

    def add_edge(self, edge: ContagionEdge) -> None:
        """Add a weighted edge (symmetric — both directions)."""
        if edge.source not in self.edges:
            self.edges[edge.source] = {}
        if edge.target not in self.edges:
            self.edges[edge.target] = {}

        self.edges[edge.source][edge.target] = edge

        # Symmetric: create reverse edge with same weight
        reverse = ContagionEdge(
            source=edge.target,
            target=edge.source,
            weight=edge.weight,
            components=dict(edge.components),
        )
        self.edges[edge.target][edge.source] = reverse

    def normalize_entity(self, name: str) -> str:
        """Resolve an entity name or alias to its canonical name."""
        return self._alias_map.get(name.lower(), name)

    def get_peers(
        self,
        entity: str,
        min_weight: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Get peers of an entity, sorted by edge weight descending.

        Returns list of (peer_name, weight) tuples.
        """
        canonical = self.normalize_entity(entity)
        neighbors = self.edges.get(canonical, {})
        peers = [
            (target, edge.weight)
            for target, edge in neighbors.items()
            if edge.weight >= min_weight
        ]
        # Sort by weight descending, then by name for stability
        peers.sort(key=lambda x: (-x[1], x[0]))
        return peers

    def get_subsector_peers(self, entity: str) -> list[str]:
        """Get same-subsector peers (convenience method)."""
        canonical = self.normalize_entity(entity)
        node = self.nodes.get(canonical)
        if node is None:
            return []
        return [
            name for name, other_node in self.nodes.items()
            if other_node.subsector == node.subsector and name != canonical
        ]

    def get_edge_weight(self, source: str, target: str) -> float:
        """Get edge weight between two entities (0.0 if no edge)."""
        src = self.normalize_entity(source)
        tgt = self.normalize_entity(target)
        edge = self.edges.get(src, {}).get(tgt)
        return edge.weight if edge is not None else 0.0


# ============================================================
# Graph Construction
# ============================================================

def load_entity_graph(
    yaml_path: Path,
    config: dict[str, Any] | None = None,
) -> EntityGraph:
    """Build an EntityGraph from nbfc_entities.yaml + contagion config.

    # 🎓 The YAML groups entities by subsector. We iterate all pairs and
    # assign edge weights: intra_subsector for same group, cross_subsector
    # for different groups. No self-edges (entity can't contagion itself).

    Args:
        yaml_path: Path to nbfc_entities.yaml
        config: contagion config dict (edge_weights section). If None, uses defaults.
    """
    if config is None:
        config = {}

    edge_weights = config.get("edge_weights", {})
    intra_weight = edge_weights.get("intra_subsector", 0.8)
    cross_weight = edge_weights.get("cross_subsector", 0.1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    graph = EntityGraph()

    # --- Step 1: Create nodes ---
    subsectors = data.get("subsectors", {})
    for subsector_name, entities in subsectors.items():
        for entity_data in entities:
            node = EntityNode(
                name=entity_data["name"],
                full_name=entity_data.get("full_name", entity_data["name"]),
                subsector=subsector_name,
                status=entity_data.get("status", "active"),
                aliases=entity_data.get("aliases", []),
            )
            graph.add_node(node)

    # --- Step 2: Create edges ---
    # All pairwise combinations (no self-edges, symmetric so we only add once)
    node_names = sorted(graph.nodes.keys())
    for i, name_a in enumerate(node_names):
        for name_b in node_names[i + 1:]:
            node_a = graph.nodes[name_a]
            node_b = graph.nodes[name_b]

            same_subsector = (node_a.subsector == node_b.subsector)
            weight = intra_weight if same_subsector else cross_weight

            # Skip zero-weight edges to keep graph sparse
            if weight <= 0:
                continue

            edge = ContagionEdge(
                source=name_a,
                target=name_b,
                weight=weight,
                components={"subsector": weight},
            )
            graph.add_edge(edge)

    n_nodes = len(graph.nodes)
    n_edges = sum(len(targets) for targets in graph.edges.values()) // 2  # symmetric
    logger.info(
        f"Built entity graph: {n_nodes} nodes, {n_edges} edges "
        f"(intra={intra_weight}, cross={cross_weight})"
    )

    return graph


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and inspect the NBFC entity contagion graph"
    )
    parser.add_argument(
        "--entities", type=Path, default=Path("configs/nbfc_entities.yaml"),
        help="Path to entity YAML",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/contagion_config.yaml"),
        help="Path to contagion config YAML",
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="Show peers for a specific entity",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    graph = load_entity_graph(args.entities, config)

    # Print summary
    print(f"\nEntity Graph: {len(graph.nodes)} nodes")
    print(f"Subsectors: {sorted(set(n.subsector for n in graph.nodes.values()))}")
    print()

    if args.entity:
        canonical = graph.normalize_entity(args.entity)
        node = graph.nodes.get(canonical)
        if node is None:
            print(f"Entity '{args.entity}' not found in graph.")
            return

        print(f"Entity: {canonical} (subsector: {node.subsector})")
        print(f"Peers (sorted by weight):")
        for peer, weight in graph.get_peers(canonical):
            peer_node = graph.nodes[peer]
            print(f"  {weight:.2f}  {peer} ({peer_node.subsector})")
    else:
        # Show subsector summary
        from collections import Counter
        subsector_counts = Counter(n.subsector for n in graph.nodes.values())
        for subsector, count in sorted(subsector_counts.items()):
            entities = [n.name for n in graph.nodes.values() if n.subsector == subsector]
            print(f"  {subsector} ({count}): {', '.join(sorted(entities))}")


if __name__ == "__main__":
    main()
