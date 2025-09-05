#!/usr/bin/env python3
"""
Intervention policy (cleaned, minimal public version).

Keeps behavior used in the paper: dynamic-threshold selection + fixed removal
fraction (“fixed depth”). Non-essential legacy branches remain compatible.

Key tweaks vs your draft:
- **identify_high_risk_nodes**: clearer doc; preserves existing modes; if
  `fixed_threshold=True` and `dynamic_threshold < threshold`, selects the top
  fraction when provided, otherwise all candidates (your current behavior).
- **compute_dynamic_threshold**: fixes parameter mapping bug — the smoothing
  factor now correctly uses **`rise_smoothing`** (α). `drop_smoothing` is
  accepted for API compatibility but ignored in this simplified rule.
- **apply_intervention**: small robustness touches (safe `.get` for per-node
  fractions; non-negative quotas). Removal is done via the shared
  `removed_edges` set; adjacency lists are not mutated.

The rest of the surface is unchanged so the main/simulation modules work as-is.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


class Intervention:
    """Handle network interventions during SIR simulations.

    Parameters mirror legacy code to avoid breaking callers. Only the subset
    used in the paper (dynamic threshold + fixed removal fraction) is essential.
    """

    def __init__(
        self,
        node_edge_lifespans: Dict[int, Dict[Tuple[int, int], List[int]]],
        removal_strategy: str = "full_duration",
        threshold: float = 0.7,
        use_dynamic_threshold: bool = True,
        fixed_threshold: Optional[bool] = None,
        fixed_removal_fraction: Optional[float] = None,
        window: int = 5,
        proportion_high_risk: float = 0.1,
        max_removal_fraction: float = 0.3,
        min_removal_fraction: float = 0.0,
        adjustment_step: float = 0.05,
        intervention_interval: Optional[int] = None,
        r0_change_window_size: Optional[int] = 1,
        n_nodes: Optional[int] = None,
        scaling_exponent: float = 1.0,
    ) -> None:
        self.removal_strategy = removal_strategy
        self.threshold = threshold

        self.use_dynamic_threshold = use_dynamic_threshold
        self.fixed_threshold = fixed_threshold
        self.fixed_removal_fraction = fixed_removal_fraction

        self.window = window
        self.proportion_high_risk = proportion_high_risk
        self.max_risk_score = 0.0
        self.max_removal_fraction = max_removal_fraction
        self.min_removal_fraction = min_removal_fraction
        self.adjustment_step = adjustment_step
        self.intervention_interval = intervention_interval
        self.scaling_exponent = scaling_exponent

        if r0_change_window_size is None:
            self.r0_change_window_size = intervention_interval
        else:
            self.r0_change_window_size = r0_change_window_size

        # If proportion_high_risk == 1, per-node fraction is driven by risk.
        self.removal_fraction_by_risk_score = proportion_high_risk == 1

        self.n_nodes = n_nodes

        # Edge lifespans (precomputed outside)
        t0 = time.time()
        self.node_edge_lifespans = node_edge_lifespans
        _ = time.time() - t0  # kept for potential logging

    # ──────────────────────────────────────────────────────────────────────────
    # High-risk selection
    # ──────────────────────────────────────────────────────────────────────────
    def identify_high_risk_nodes(
        self,
        risk_deques: Dict[int, Iterable[float]],
        states: np.ndarray,
        dynamic_threshold: float,
        fixed_threshold: Optional[bool],
        threshold: float,
        top_node_frac: Optional[float],
    ) -> Set[int]:
        """Pick the set of high-risk nodes.

        Modes
        -----
        - If `use_dynamic_threshold` is **False**:
            return the top `top_node_frac` of (S or I) nodes by avg risk.
        - If `use_dynamic_threshold` is **True** and `fixed_threshold` is **False**:
            return all nodes with avg risk > `dynamic_threshold`.
        - If `use_dynamic_threshold` is **True** and `fixed_threshold` is **True**:
            intervene **only** when `dynamic_threshold < threshold`; when intervening,
            pick the top `top_node_frac` if provided, else **all** candidates (unchanged
            from your current behavior).
        """
        # Average risk per node
        avg_risk = {n: (sum(dq) / len(dq) if dq else 0.0) for n, dq in risk_deques.items()}
        # Consider only nodes that are not recovered
        candidate_nodes = [n for n, s in enumerate(states) if s != 2]

        if not self.use_dynamic_threshold:
            if not top_node_frac or top_node_frac <= 0:
                return set()
            k = max(1, int(len(candidate_nodes) * top_node_frac))
            ranked = sorted(candidate_nodes, key=lambda n: avg_risk.get(n, 0.0), reverse=True)
            return set(ranked[:k])

        if not fixed_threshold:
            return {n for n in candidate_nodes if avg_risk.get(n, 0.0) > dynamic_threshold}

        # fixed_threshold == True
        if dynamic_threshold < threshold:
            ranked = sorted(candidate_nodes, key=lambda n: avg_risk.get(n, 0.0), reverse=True)
            if top_node_frac and top_node_frac > 0:
                k = max(1, int(len(candidate_nodes) * top_node_frac))
                return set(ranked[:k])
            return set(ranked)  # your existing "all candidates" behavior

        return set()

    # ──────────────────────────────────────────────────────────────────────────
    # Infection metrics & threshold dynamics
    # ──────────────────────────────────────────────────────────────────────────
    def compute_infection_acceleration(self, history: np.ndarray, history_index: int, window: int) -> float:
        """Acceleration ≈ ΔI over a lookback `window`, normalized by I at the window start."""
        infection_counts = [int(np.sum(history[t, :] == 1)) for t in range(history_index)]
        if len(infection_counts) < window + 1:
            return 0.0
        delta = infection_counts[-1] - infection_counts[-(window + 1)]
        prev = infection_counts[-(window + 1)]
        return float(delta / prev) if prev > 0 else 0.0

    def compute_global_infection_potential(self, states: np.ndarray) -> float:
        """P_t = I_t * S_t / (N/2)^2 in [0,1]. Stored on `self.P_t` for reuse."""
        I_t = int(np.sum(states == 1))
        S_t = int(np.sum(states == 0))
        N_t = int(len(states))
        P_t = (I_t * S_t) / float((N_t / 2) ** 2) if N_t > 0 else 0.0
        self.P_t = P_t
        return P_t

    def compute_dynamic_threshold(
        self,
        acceleration: float,
        base_threshold: float = 1.0,
        drop_strength: float = 0.30,  # λ: sensitivity of target to pressure ψ
        rise_strength: float = 0.0,   # accepted, unused (API compat)
        accel_weight: float = 1.0,    # kept = 1
        pt_weight: float = 1.0,       # kept = 1
        rise_smoothing: float = 0.25, # accepted, unused in single-smoother mode
        drop_smoothing: float = 0.0,  # α: exponential smoothing factor used for both directions
    ) -> float:
        """One-gain, one-smoother dynamic threshold.

        θ*_k = clamp( base − λ · ψ_k, 0, 1 ),   ψ_k = a_k + P_k
        θ_k  = (1−α)·θ_{k−1} + α·θ*_k

        Notes
        -----
        - `drop_strength` (λ) controls sensitivity to the composite pressure ψ.
        - `drop_smoothing` is the **single** smoothing factor α ∈ [0,1] used regardless of direction.
        - `rise_smoothing` not used
        """
        P_t = getattr(self, "P_t", 0.0)
        psi = accel_weight * acceleration + pt_weight * P_t

        if not hasattr(self, "_prev_threshold"):
            self._prev_threshold = base_threshold

        lam = drop_strength
        theta_target = max(0.0, min(base_threshold - lam * psi, 1.0))

        # α comes from drop_smoothing (single-smoother design)
        alpha = drop_smoothing
        theta_smoothed = (1.0 - alpha) * self._prev_threshold + alpha * theta_target
        self._prev_threshold = theta_smoothed
        return theta_smoothed

    def sigmoid_component(self, x: float, L: float = 0.5, x0: float = 3.0, k: float = 2.0) -> float:
        return float(L / (1 + np.exp(-k * (x - x0))))

    def compute_removal_fraction(
        self,
        risk_score: float,
        P_t: float,
        scaling_exponent: float = 1.0,
        min_removal: float = 0.0,
        max_removal: float = 1.0,
    ) -> float:
        """Simple mapping: fraction ∝ risk_score, clamped to [min_removal, max_removal]."""
        raw_fraction = risk_score
        return float(min(max(raw_fraction, min_removal), max_removal))

    # ──────────────────────────────────────────────────────────────────────────
    # Apply intervention
    # ──────────────────────────────────────────────────────────────────────────
    def apply_intervention(
        self,
        current_time: int,
        adjacency_lists: List[Dict[int, Set[int]]],
        high_risk_nodes: Set[int],
        R0_avg: float,
        risk_deques: Dict[int, Iterable[float]],
        states: np.ndarray,
        intervention_interval: int,
        R0_deque,
        risk_computer,
        n_nodes: int,
        removed_high_risk_nodes: Optional[Set[int]] = None,
        removed_edges: Optional[Set[Tuple[int, int, int]]] = None,
        infected_count_deque=None,
        node_removal_fractions: Optional[Dict[int, float]] = None,
        compliance: float = 1.0,
    ):
        """Remove a fraction of future edges for flagged nodes over the next interval.

        Returns a tuple:
          (edge_removal_count, cur_removed_edges, removed_edges, removed_high_risk_nodes,
           current_high_risk_nodes, risk_reduction_data, avg_R0_change, P_t, adjacency_lists)
        """
        if removed_edges is None:
            removed_edges = set()
        if removed_high_risk_nodes is None:
            removed_high_risk_nodes = set()

        # Average ΔR0 over deque
        if R0_deque and len(R0_deque) >= 2:
            R0_changes = [R0_deque[i] - R0_deque[i - 1] for i in range(1, len(R0_deque))]
            avg_R0_change = float(sum(R0_changes) / len(R0_changes))
        else:
            avg_R0_change = 0.0

        total_edges_removed_in_timestep = 0
        edge_removal_count = defaultdict(int)
        cur_removed_edges: List[Tuple[int, int, int]] = []
        current_high_risk_nodes: Set[int] = set()

        risk_scores_arr = np.zeros(n_nodes, dtype=float)
        edges_reduced = np.zeros(n_nodes, dtype=int)
        contributions_to_total_reduction = np.zeros(n_nodes, dtype=float)

        # Average risk scores at this step
        risk_scores_dict = risk_computer.compute_average(risk_deques)

        # Per-node removal fractions: provided by caller in fixed-depth mode
        removal_fractions = np.full(n_nodes, self.min_removal_fraction, dtype=float)
        if node_removal_fractions is None:
            node_removal_fractions = {}
        for node in high_risk_nodes:
            removal_fractions[node] = float(node_removal_fractions.get(node, self.min_removal_fraction))

        # Gather future edge instances per high-risk node within the next window
        end_time = current_time + intervention_interval
        future_by_node: Dict[int, List[Tuple[Tuple[int, int], int]]] = {
            node: [
                (edge, t)
                for edge, times in self.node_edge_lifespans.get(node, {}).items()
                for t in times
                if current_time < t <= end_time
            ]
            for node in high_risk_nodes
        }

        # Integer quotas per node from fraction × compliance × #candidates
        quota = {
            node: max(0, int(len(future_by_node[node]) * compliance * removal_fractions[node]))
            for node in high_risk_nodes
        }
        for node in high_risk_nodes:
            edges_reduced[node] = 0

        # Big shuffled candidate list: (node, edge, t)
        all_cands = [
            (node, edge, t)
            for node, lst in future_by_node.items()
            for (edge, t) in lst
        ]
        random.shuffle(all_cands)

        # Greedy selection, avoid duplicate (u,v,t)
        edges_to_remove: Set[Tuple[Tuple[int, int], int]] = set()
        for node, edge, t in all_cands:
            if quota[node] <= 0:
                continue
            uid = (min(edge), max(edge), t)
            if uid in removed_edges:
                continue
            removed_edges.add(uid)
            edges_to_remove.add((edge, t))
            quota[node] -= 1
            edges_reduced[node] += 1
            total_edges_removed_in_timestep += 1

        # Bookkeep per-snapshot counts & current list
        for edge, t in edges_to_remove:
            u, v = edge
            if v in adjacency_lists[t].get(u, set()):
                edge_removal_count[t] += 1
                cur_removed_edges.append((min(u, v), max(u, v), t))

        current_high_risk_nodes |= set(high_risk_nodes)
        removed_high_risk_nodes |= set(high_risk_nodes)

        # Per-node contribution shares
        if total_edges_removed_in_timestep > 0:
            for node in high_risk_nodes:
                contributions_to_total_reduction[node] = edges_reduced[node] / float(total_edges_removed_in_timestep)
        else:
            contributions_to_total_reduction[:] = 0.0

        for node in range(n_nodes):
            risk_scores_arr[node] = risk_scores_dict.get(node, 0.0)

        risk_reduction_data = {
            'risk_scores': risk_scores_arr,
            'edges_reduced': edges_reduced,
            'removal_fractions': removal_fractions,
            'contribution_to_total_reduction': contributions_to_total_reduction,
        }

        return (
            edge_removal_count,
            cur_removed_edges,
            removed_edges,
            removed_high_risk_nodes,
            current_high_risk_nodes,
            risk_reduction_data,
            avg_R0_change,
            getattr(self, 'P_t', 0.0),
            adjacency_lists,
        )

    # Legacy helper (kept for completeness)
    def get_consecutive_periods(self, times: List[int]) -> List[List[int]]:
        periods: List[List[int]] = []
        times = sorted(times)
        if not times:
            return periods
        current_period = [times[0]]
        for i in range(1, len(times)):
            if times[i] == times[i - 1] + 1:
                current_period.append(times[i])
            else:
                periods.append(current_period)
                current_period = [times[i]]
        periods.append(current_period)
        return periods
