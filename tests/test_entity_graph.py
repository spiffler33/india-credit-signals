# WHY THIS: Unit tests for the entity contagion graph using synthetic YAML fixtures.
# Tests verify edge weight logic, alias resolution, peer queries, and graph structure
# without depending on the real nbfc_entities.yaml file (which may change over time).

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.signals.entity_graph import (
    ContagionEdge,
    EntityGraph,
    EntityNode,
    load_entity_graph,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mini_yaml(tmp_path: Path) -> Path:
    """Create a minimal nbfc_entities.yaml with 2 subsectors, 5 entities."""
    data = {
        "subsectors": {
            "housing_finance": [
                {
                    "name": "DHFL",
                    "full_name": "Dewan Housing Finance Corporation Limited",
                    "aliases": ["DHFL", "Dewan Housing"],
                    "status": "defaulted",
                },
                {
                    "name": "Indiabulls Housing Finance",
                    "full_name": "Indiabulls Housing Finance Limited",
                    "aliases": ["Indiabulls Housing", "IBHFL"],
                    "status": "active",
                },
                {
                    "name": "PNB Housing Finance",
                    "full_name": "PNB Housing Finance Limited",
                    "aliases": ["PNB Housing"],
                    "status": "active",
                },
            ],
            "diversified_nbfc": [
                {
                    "name": "Cholamandalam Investment",
                    "full_name": "Cholamandalam Investment and Finance Company Limited",
                    "aliases": ["Chola", "Cholamandalam"],
                    "status": "active",
                },
                {
                    "name": "Bajaj Finance",
                    "full_name": "Bajaj Finance Limited",
                    "aliases": ["Bajaj Finance", "BFL"],
                    "status": "active",
                },
            ],
        }
    }
    yaml_path = tmp_path / "entities.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    return yaml_path


@pytest.fixture
def default_config() -> dict:
    """Default contagion config with standard weights."""
    return {
        "edge_weights": {
            "intra_subsector": 0.8,
            "cross_subsector": 0.1,
        }
    }


@pytest.fixture
def graph(mini_yaml: Path, default_config: dict) -> EntityGraph:
    """Pre-built graph from mini YAML."""
    return load_entity_graph(mini_yaml, default_config)


# ============================================================
# Test: Graph Construction
# ============================================================

class TestGraphConstruction:
    def test_node_count(self, graph: EntityGraph) -> None:
        assert len(graph.nodes) == 5

    def test_subsector_assignment(self, graph: EntityGraph) -> None:
        assert graph.nodes["DHFL"].subsector == "housing_finance"
        assert graph.nodes["Cholamandalam Investment"].subsector == "diversified_nbfc"

    def test_intra_subsector_edge_weight(self, graph: EntityGraph) -> None:
        """DHFL → Indiabulls = same subsector = 0.8."""
        weight = graph.get_edge_weight("DHFL", "Indiabulls Housing Finance")
        assert weight == 0.8

    def test_cross_subsector_edge_weight(self, graph: EntityGraph) -> None:
        """DHFL → Cholamandalam = different subsector = 0.1."""
        weight = graph.get_edge_weight("DHFL", "Cholamandalam Investment")
        assert weight == 0.1

    def test_no_self_edges(self, graph: EntityGraph) -> None:
        """No entity should have an edge to itself."""
        for name in graph.nodes:
            assert name not in graph.edges.get(name, {}), f"Self-edge found for {name}"

    def test_symmetric_edges(self, graph: EntityGraph) -> None:
        """Edge A→B should equal edge B→A."""
        w_ab = graph.get_edge_weight("DHFL", "Indiabulls Housing Finance")
        w_ba = graph.get_edge_weight("Indiabulls Housing Finance", "DHFL")
        assert w_ab == w_ba

    def test_components_dict(self, graph: EntityGraph) -> None:
        """Edge should have a components dict for v2 extensibility."""
        edge = graph.edges["DHFL"]["Indiabulls Housing Finance"]
        assert "subsector" in edge.components
        assert edge.components["subsector"] == 0.8

    def test_zero_cross_weight_skips_edges(self, mini_yaml: Path) -> None:
        """When cross_subsector = 0, no cross-subsector edges should exist."""
        config = {"edge_weights": {"intra_subsector": 0.8, "cross_subsector": 0.0}}
        g = load_entity_graph(mini_yaml, config)
        # DHFL → Cholamandalam should have no edge
        assert g.get_edge_weight("DHFL", "Cholamandalam Investment") == 0.0


# ============================================================
# Test: Alias Resolution
# ============================================================

class TestAliasResolution:
    def test_canonical_name(self, graph: EntityGraph) -> None:
        assert graph.normalize_entity("DHFL") == "DHFL"

    def test_alias_resolves(self, graph: EntityGraph) -> None:
        assert graph.normalize_entity("Dewan Housing") == "DHFL"

    def test_case_insensitive(self, graph: EntityGraph) -> None:
        assert graph.normalize_entity("dhfl") == "DHFL"
        assert graph.normalize_entity("dewan housing") == "DHFL"

    def test_unknown_entity_returns_itself(self, graph: EntityGraph) -> None:
        assert graph.normalize_entity("Unknown Corp") == "Unknown Corp"

    def test_alias_chola(self, graph: EntityGraph) -> None:
        """Cholamandalam has alias 'Cholamandalam' (without 'Investment')."""
        assert graph.normalize_entity("Cholamandalam") == "Cholamandalam Investment"
        assert graph.normalize_entity("Chola") == "Cholamandalam Investment"

    def test_alias_ibhfl(self, graph: EntityGraph) -> None:
        assert graph.normalize_entity("IBHFL") == "Indiabulls Housing Finance"


# ============================================================
# Test: Peer Queries
# ============================================================

class TestPeerQueries:
    def test_get_peers_sorted_by_weight(self, graph: EntityGraph) -> None:
        """Peers should be sorted by weight descending."""
        peers = graph.get_peers("DHFL")
        # First two should be intra-subsector (0.8), then cross-subsector (0.1)
        weights = [w for _, w in peers]
        assert weights == sorted(weights, reverse=True)
        assert peers[0][1] == 0.8  # intra-subsector first
        assert peers[-1][1] == 0.1  # cross-subsector last

    def test_get_peers_min_weight_filter(self, graph: EntityGraph) -> None:
        """min_weight should filter out low-weight edges."""
        peers = graph.get_peers("DHFL", min_weight=0.5)
        # Only intra-subsector peers (weight 0.8)
        assert all(w >= 0.5 for _, w in peers)
        assert len(peers) == 2  # Indiabulls + PNB Housing

    def test_get_subsector_peers(self, graph: EntityGraph) -> None:
        """Same-subsector peers for DHFL = Indiabulls + PNB Housing."""
        peers = graph.get_subsector_peers("DHFL")
        assert set(peers) == {"Indiabulls Housing Finance", "PNB Housing Finance"}

    def test_get_subsector_peers_via_alias(self, graph: EntityGraph) -> None:
        """Should work with alias input."""
        peers = graph.get_subsector_peers("Dewan Housing")
        assert "Indiabulls Housing Finance" in peers

    def test_get_peers_unknown_entity(self, graph: EntityGraph) -> None:
        """Unknown entity should return empty list."""
        peers = graph.get_peers("Unknown Corp")
        assert peers == []

    def test_get_subsector_peers_unknown(self, graph: EntityGraph) -> None:
        assert graph.get_subsector_peers("Unknown Corp") == []


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Empty subsectors dict should produce empty graph."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text(yaml.dump({"subsectors": {}}))
        g = load_entity_graph(yaml_path, {})
        assert len(g.nodes) == 0

    def test_single_entity_subsector(self, tmp_path: Path) -> None:
        """Subsector with one entity has no intra-subsector edges."""
        data = {
            "subsectors": {
                "solo": [
                    {"name": "Solo Corp", "full_name": "Solo Corp", "aliases": [], "status": "active"},
                ],
            }
        }
        yaml_path = tmp_path / "solo.yaml"
        yaml_path.write_text(yaml.dump(data))
        g = load_entity_graph(yaml_path, {})
        assert len(g.nodes) == 1
        assert g.get_peers("Solo Corp") == []

    def test_default_config_when_none(self, mini_yaml: Path) -> None:
        """Passing config=None should use defaults (0.8 / 0.1)."""
        g = load_entity_graph(mini_yaml, config=None)
        assert g.get_edge_weight("DHFL", "Indiabulls Housing Finance") == 0.8
        assert g.get_edge_weight("DHFL", "Cholamandalam Investment") == 0.1
