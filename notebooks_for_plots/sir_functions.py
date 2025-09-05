#!/usr/bin/env python3
"""
SIR helpers + simple risk-precompute utilities.


"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Iterable

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# R0 proxies
# ──────────────────────────────────────────────────────────────────────────────

def compute_R0(G) -> float:
    """Compute an R0 proxy from a static networkx.Graph `G`.

    R0 ≈ (E[k²] − E[k]) / E[k]. Returns 0.0 if the mean degree is 0.
    """
    degrees = np.array([deg for _, deg in G.degree()])
    mean_k = degrees.mean() if degrees.size else 0.0
    if mean_k == 0.0:
        return 0.0
    mean_k2 = (degrees ** 2).mean()
    return float((mean_k2 - mean_k) / mean_k)


def compute_R0_from_adj_list(
    adj_list: Dict[int, Iterable[int]],
    removed_edges: Set[Tuple[int, int, int]],
    current_time: int,
) -> float:
    """Compute the R0 proxy using an adjacency list at a given time, ignoring removed edges."""
    degs: List[int] = []
    for node, neighbors in adj_list.items():
        active_neighbors = [
            nbr
            for nbr in neighbors
            if (min(node, nbr), max(node, nbr), current_time) not in removed_edges
        ]
        degs.append(len(active_neighbors))

    if not degs:
        return 0.0
    degrees = np.array(degs, dtype=np.float64)
    mean_k = degrees.mean()
    if mean_k == 0.0:
        return 0.0
    mean_k2 = (degrees ** 2).mean()
    return float((mean_k2 - mean_k) / mean_k)


# ──────────────────────────────────────────────────────────────────────────────
# Core SIR generator
# ──────────────────────────────────────────────────────────────────────────────

def sir_simulation(
    initial_infected_nodes: Iterable[int],
    adjacency_lists: List[Dict[int, Set[int]]],
    start_t: int,
    beta: float,
    gamma: float,
    n_timesteps: int,
    states: np.ndarray,
    removed_edges: Set[Tuple[int, int, int]],
):
    """Yield SIR states for `n_timesteps`, starting at absolute time `start_t`.

    `states` is modified in-place; each yield returns a snapshot copy.
    States: 0=S, 1=I, 2=R.
    """
    if start_t == 0:
        for rn in initial_infected_nodes:
            states[rn] = 1  # Infect initial nodes

    infected_indices: Set[int] = set(np.where(states == 1)[0])

    for t in range(start_t, start_t + n_timesteps):
        current_time = t % len(adjacency_lists)
        adj_list = adjacency_lists[current_time]

        next_states = states.copy()
        new_infected_indices: Set[int] = set()

        # Recovery + infection
        for node in infected_indices:
            # Recovery
            if np.random.random() < gamma:
                next_states[node] = 2  # R
                continue

            # Infection attempts
            for nbr in adj_list.get(node, ()):  # neighbors may be set or list
                if states[nbr] == 0:  # susceptible
                    edge = (min(node, nbr), max(node, nbr), current_time)
                    if edge not in removed_edges and np.random.random() < beta:
                        next_states[nbr] = 1
                        new_infected_indices.add(nbr)
            new_infected_indices.add(node)  # stays infected if not recovered this tick

        states = next_states
        infected_indices = new_infected_indices
        yield states


# ──────────────────────────────────────────────────────────────────────────────
# Precomputations
# ──────────────────────────────────────────────────────────────────────────────

def precompute_edge_lifespans(temporal_network) -> Tuple[dict, dict]:
    """Return (edge_lifespans, node_edge_lifespans) for a temporal network (list of graphs)."""
    from collections import defaultdict as dd

    edge_lifespans = dd(list)
    for time_step, G in enumerate(temporal_network):
        for edge in G.edges():
            e = tuple(sorted(edge))
            edge_lifespans[e].append(time_step)

    node_edge_lifespans = dd(dict)
    for edge, times in edge_lifespans.items():
        u, v = edge
        node_edge_lifespans[u][edge] = times
        node_edge_lifespans[v][edge] = times

    return edge_lifespans, node_edge_lifespans


def _dd_int() -> defaultdict:
    return defaultdict(int)


def precompute_degrees(adjacency_lists: List[Dict[int, Set[int]]]) -> Dict[int, Dict[int, int]]:
    """Return precomputed degree per node per timestep: degs[node][t] = degree."""
    degs: Dict[int, Dict[int, int]] = defaultdict(_dd_int)
    for t, adj in enumerate(adjacency_lists):
        for node, neighbors in adj.items():
            degs[node][t] = len(neighbors)
    return degs


# ──────────────────────────────────────────────────────────────────────────────
# Risk precompute helpers (kept for compatibility; only degree is used in paper)
# ──────────────────────────────────────────────────────────────────────────────

def compute_max_ninl4_risk(temporal_network, precomputed_degrees) -> float:
    print("Compute max NINL4 risk in advance…")
    max_risk = 0.0
    nodes = list(precomputed_degrees.keys())
    for t, G in enumerate(temporal_network):
        adj = {n: set(G.neighbors(n)) for n in nodes}
        degree = {n: precomputed_degrees[n].get(t, 0) for n in nodes}

        ninl0 = {n: degree[n] + sum(degree.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl3 = {n: sum(ninl2.get(v, 0) for v in adj.get(n, ())) for n in nodes}

        snapshot_max = max(ninl3.values(), default=0.0)
        if snapshot_max > max_risk:
            max_risk = snapshot_max
    print(f"Max NINL4 risk: {max_risk}")
    return float(max_risk)


def compute_max_ninl3_risk(temporal_network, precomputed_degrees) -> float:
    print("Compute max NINL3 risk in advance…")
    max_risk = 0.0
    nodes = list(precomputed_degrees.keys())
    for t, G in enumerate(temporal_network):
        adj = {n: set(G.neighbors(n)) for n in nodes}
        degree = {n: precomputed_degrees[n].get(t, 0) for n in nodes}
        ninl0 = {n: degree[n] + sum(degree.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        snapshot_max = max(ninl2.values(), default=0.0)
        if snapshot_max > max_risk:
            max_risk = snapshot_max
    print(f"Max NINL3 risk: {max_risk}")
    return float(max_risk)


def compute_max_ninl2_risk(temporal_network, precomputed_degrees) -> float:
    print("Compute max NINL2 risk in advance…")
    max_risk = 0.0
    nodes = list(precomputed_degrees.keys())
    for t, G in enumerate(temporal_network):
        adj = {n: set(G.neighbors(n)) for n in nodes}
        degree = {n: precomputed_degrees[n].get(t, 0) for n in nodes}
        ninl0 = {n: degree[n] + sum(degree.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, ())) for n in nodes}
        snapshot_max = max(ninl1.values(), default=0.0)
        if snapshot_max > max_risk:
            max_risk = snapshot_max
    print(f"Max NINL2 risk: {max_risk}")
    return float(max_risk)


def compute_max_degree_risk(temporal_network, precomputed_degrees) -> float:
    print("Compute max degree-based risk in advance…")
    max_risk = 0.0
    nodes = list(precomputed_degrees.keys())
    for t, _G in enumerate(temporal_network):
        deg_t = {n: precomputed_degrees[n].get(t, 0) for n in nodes}
        snapshot_max = max(deg_t.values(), default=0.0)
        if snapshot_max > max_risk:
            max_risk = snapshot_max
    print(f"Max degree-based risk: {max_risk}")
    return float(max_risk)


def compute_max_erm_risk(temporal_network, precomputed_degrees) -> float:
    print("Compute max ERM-based risk in advance…")
    max_risk = 0.0
    nodes = list(precomputed_degrees.keys())

    for t, G in enumerate(temporal_network):
        adj = {n: set(G.neighbors(n)) for n in nodes}
        degree = {n: precomputed_degrees[n].get(t, 0) for n in nodes}

        d1, d2, E1, E2, EC, SI = {}, {}, {}, {}, {}, {}
        for n1 in nodes:
            friends = adj.get(n1, ())
            d1[n1] = sum(degree.get(n2, 0) for n2 in friends)
        for n1 in nodes:
            friends = adj.get(n1, ())
            d2[n1] = sum(d1.get(n2, 0) for n2 in friends)
        for n1 in nodes:
            friends = adj.get(n1, ())
            if d1[n1] > 0:
                E1[n1] = -sum(
                    (degree[n2] / d1[n1]) * math.log(degree[n2] / d1[n1])
                    for n2 in friends
                    if degree[n2] > 0
                )
            else:
                E1[n1] = 0.0
            if d2[n1] > 0:
                E2[n1] = -sum(
                    (d1.get(n2, 0) / d2[n1]) * math.log(d1.get(n2, 0) / d2[n1])
                    for n2 in friends
                    if d1.get(n2, 0) > 0
                )
            else:
                E2[n1] = 0.0
        max_E2 = max(E2.values()) if E2 else 0.0
        for n1 in nodes:
            lam = (E2[n1] / max_E2) if max_E2 > 0 else 0.0
            friends = adj.get(n1, ())
            EC[n1] = sum(E1.get(n2, 0.0) + lam * E2.get(n2, 0.0) for n2 in friends)
        for n1 in nodes:
            friends = adj.get(n1, ())
            SI[n1] = sum(EC.get(n2, 0.0) for n2 in friends)

        snapshot_max = max(SI.values(), default=0.0)
        if snapshot_max > max_risk:
            max_risk = snapshot_max

    print(f"Max ERM risk: {max_risk}")
    return float(max_risk)
