# risk_computation.py

from abc import ABC, abstractmethod
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import math
from math import log1p


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class RiskComputation(ABC):
    """
    Abstract base class for risk computation methods.

    Parameters
    ----------
    normalize_risk_score : bool
        If True, normalize scores using a fixed cap (log1p-based).
    precomputed_max_risk : float | None
        If provided, used once to set a fixed normalization cap across all timesteps.
    """

    def __init__(self, normalize_risk_score: bool = False,
                 precomputed_max_risk: Optional[float] = None):
        self.global_max_risk = 0.0
        self.normalize_risk_score = normalize_risk_score
        self.precomputed_max_risk = precomputed_max_risk
        self.precomputed_max_risk_log: Optional[float] = None

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize risk structures (usually per-node deques)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, risk_deques, adjacency_lists, current_time, intervention_interval, **kwargs):
        """Update risk structures based on network evolution."""
        raise NotImplementedError

    @abstractmethod
    def compute_average(self, risk_deques):
        """Compute average risk per node from the deques."""
        raise NotImplementedError

    def normalize_risks_global(self, risk_deques: Dict[int, deque]) -> Dict[int, deque]:
        """
        Normalize risk scores in each deque using:

            norm = log1p(risk) / log1p(max_risk_cap)

        The cap is fixed once (prefer `precomputed_max_risk` if provided; else
        computed from the current values on first call) so that normalization is
        stable/comparable across time.
        """
        if self.precomputed_max_risk_log is None:
            if self.precomputed_max_risk is not None and self.precomputed_max_risk > 0:
                self.precomputed_max_risk_log = log1p(self.precomputed_max_risk)
            else:
                # Fallback: infer from current risks if no precomputed cap is given.
                all_risks = [x for dq in risk_deques.values() for x in dq]
                max_val = max(all_risks) if all_risks else 1.0
                self.precomputed_max_risk_log = log1p(max_val)

        cap = self.precomputed_max_risk_log
        return {
            node: deque([log1p(val) / cap for val in dq], maxlen=len(dq))
            for node, dq in risk_deques.items()
        }

    def normalize_values_dict(self, raw_values: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize a dict of scalar values the same way as deques:
            norm = log1p(val) / log1p(max_risk_cap)
        """
        if self.precomputed_max_risk_log is None:
            if self.precomputed_max_risk is not None and self.precomputed_max_risk > 0:
                self.precomputed_max_risk_log = log1p(self.precomputed_max_risk)
            else:
                self.precomputed_max_risk_log = log1p(max(raw_values.values(), default=1.0))

        cap = self.precomputed_max_risk_log
        out: Dict[int, float] = {}
        for node, val in raw_values.items():
            norm = log1p(val) / cap
            if norm > 1:  # diagnostic only; behavior unchanged
                print(f"[WARN] Normalized value > 1 for node {node}: "
                      f"raw={val}, log1p(raw)={log1p(val)}, cap={cap}, norm={norm}")
            out[node] = norm
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Degree-based risk
# ──────────────────────────────────────────────────────────────────────────────

class DegreeBasedRiskComputation(RiskComputation):
    """
    Degree‐based risk (exact *live* degrees, honoring removed edges).
    """

    def __init__(self,
                 normalize_risk_score: bool = False,
                 precomputed_max_risk: Optional[float] = None):
        super().__init__(normalize_risk_score, precomputed_max_risk)

    def initialize(self,
                   precomputed_degrees: Dict[int, Dict[int, int]],
                   adjacency_lists: List[Dict[int, Set[int]]],
                   maxlen: int = 5):
        # Keep exact degree at t=0 from adjacency_lists[0]
        self.adjacency_lists = adjacency_lists
        self.n_nodes = len(precomputed_degrees)
        nodes = list(precomputed_degrees.keys())

        adj0 = adjacency_lists[0]
        deg0 = {n: len(adj0.get(n, ())) for n in nodes}

        dq = {n: deque([float(deg0[n])], maxlen=maxlen) for n in nodes}
        return self.normalize_risks_global(dq) if self.normalize_risk_score else dq

    def update(self,
               risk_deques: Dict[int, deque],
               adjacency_lists: List[Dict[int, Set[int]]],
               current_time: int,
               intervention_interval: int,
               removed_edges: Set[Tuple[int, int, int]],
               n_past: int = 2):
        start_t = max(0, current_time - intervention_interval)
        nodes = range(self.n_nodes)
        cache: Dict[int, List[float]] = defaultdict(list)

        for t in range(start_t, current_time):
            ts = t % len(adjacency_lists)

            # live adjacency (strip removed edges at this snapshot)
            adj_live = {u: set(adjacency_lists[ts].get(u, ())) for u in nodes}
            for (u, v, t_removed) in removed_edges:
                if t_removed == ts:
                    adj_live[u].discard(v)
                    adj_live[v].discard(u)

            deg_live = {u: float(len(adj_live[u])) for u in nodes}
            for u in nodes:
                cache[u].append(deg_live[u])

        for u in nodes:
            if cache[u]:
                mean_deg = float(np.mean(cache[u]))
                dq = risk_deques[u]
                dq.append(mean_deg)
                if len(dq) > n_past:
                    dq.popleft()

        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def compute_average(self, risk_deques: Dict[int, deque]) -> Dict[int, float]:
        return {u: sum(dq) / len(dq) for u, dq in risk_deques.items()}


# ──────────────────────────────────────────────────────────────────────────────
# NINL (layers 4, 3, 2) risks
# ──────────────────────────────────────────────────────────────────────────────

class Ninl4RiskComputation(RiskComputation):
    """NINL-layer-4 risk using precomputed degrees."""

    def __init__(self, normalize_risk_score: bool = False, precomputed_max_risk: Optional[float] = None):
        super().__init__(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)

    def initialize(self, precomputed_degrees, adjacency_lists, maxlen: int = 5):
        self.precomputed_degrees = precomputed_degrees
        self.adjacency_lists = adjacency_lists
        nodes = list(self.precomputed_degrees.keys())
        time_step = 0

        degree_dict = {n: self.precomputed_degrees[n].get(time_step, 0) for n in nodes}
        adj = self.adjacency_lists[time_step]

        ninl0 = {n: degree_dict[n] + sum(degree_dict.get(v, 0) for v in adj.get(n, [])) for n in nodes}
        ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, [])) for n in nodes}
        ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, [])) for n in nodes}
        ninl3 = {n: sum(ninl2.get(v, 0) for v in adj.get(n, [])) for n in nodes}

        risk_deques = {n: deque([ninl3[n]], maxlen=maxlen) for n in nodes}
        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def update(self, risk_deques, adjacency_lists, current_time, intervention_interval, removed_edges, n_past: int = 2):
        start_time = max(0, current_time - intervention_interval)
        nodes = list(self.precomputed_degrees.keys())
        risk_cache: Dict[int, List[float]] = defaultdict(list)

        for t in range(start_time, current_time):
            ts = t % len(self.adjacency_lists)
            base_adj = self.adjacency_lists[ts]

            # adjusted adjacency
            adj = {}
            for n in nodes:
                nbrs = set(base_adj.get(n, []))
                for v in list(nbrs):
                    if (min(n, v), max(n, v), ts) in removed_edges:
                        nbrs.discard(v)
                adj[n] = list(nbrs)

            degree = {n: self.precomputed_degrees[n].get(ts, 0) for n in nodes}

            # account degree decrements from removed edges at this ts
            deg_adj = defaultdict(int)
            for u, v, ts_rm in removed_edges:
                if ts_rm == ts:
                    if u in degree: deg_adj[u] += 1
                    if v in degree: deg_adj[v] += 1
            for n in nodes:
                degree[n] = max(degree[n] - deg_adj.get(n, 0), 0)

            ninl0 = {n: degree[n] + sum(degree.get(v, 0) for v in adj.get(n, [])) for n in nodes}
            ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, [])) for n in nodes}
            ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, [])) for n in nodes}
            ninl3 = {n: sum(ninl2.get(v, 0) for v in adj.get(n, [])) for n in nodes}

            for n in nodes:
                risk_cache[n].append(ninl3[n])

        for n in nodes:
            if risk_cache[n]:
                avg = float(np.mean(risk_cache[n]))
                dq = risk_deques[n]
                dq.append(avg)
                if len(dq) > n_past:
                    dq.popleft()

        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def compute_average(self, risk_deques):
        return {n: sum(dq) / len(dq) for n, dq in risk_deques.items()}


class Ninl3RiskComputation(RiskComputation):
    """NINL-layer-3 risk using precomputed degrees."""

    def __init__(self, normalize_risk_score: bool = False, precomputed_max_risk: Optional[float] = None):
        super().__init__(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)

    def initialize(self, precomputed_degrees, adjacency_lists, maxlen: int = 5):
        self.precomputed_degrees = precomputed_degrees
        self.adjacency_lists = adjacency_lists
        nodes = list(self.precomputed_degrees.keys())
        time_step = 0

        degree_dict = {n: self.precomputed_degrees[n].get(time_step, 0) for n in nodes}
        adj = self.adjacency_lists[time_step]

        ninl0 = {n: degree_dict[n] + sum(degree_dict.get(v, 0) for v in adj.get(n, [])) for n in nodes}
        ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, [])) for n in nodes}
        ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, [])) for n in nodes}

        risk_deques = {n: deque([ninl2[n]], maxlen=maxlen) for n in nodes}
        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def update(self, risk_deques, adjacency_lists, current_time, intervention_interval, removed_edges, n_past: int = 2):
        start_time = max(0, current_time - intervention_interval)
        nodes = list(self.precomputed_degrees.keys())
        risk_cache: Dict[int, List[float]] = defaultdict(list)

        for t in range(start_time, current_time):
            ts = t % len(self.adjacency_lists)
            base_adj = self.adjacency_lists[ts]

            # adjusted adjacency
            adj = {}
            for n in nodes:
                nbrs = set(base_adj.get(n, []))
                for v in list(nbrs):
                    if (min(n, v), max(n, v), ts) in removed_edges:
                        nbrs.discard(v)
                adj[n] = list(nbrs)

            degree = {n: self.precomputed_degrees[n].get(ts, 0) for n in nodes}

            deg_adj = defaultdict(int)
            for u, v, ts_rm in removed_edges:
                if ts_rm == ts:
                    if u in degree: deg_adj[u] += 1
                    if v in degree: deg_adj[v] += 1
            for n in nodes:
                degree[n] = max(degree[n] - deg_adj.get(n, 0), 0)

            ninl0 = {n: degree[n] + sum(degree.get(v, 0) for v in adj.get(n, [])) for n in nodes}
            ninl1 = {n: sum(ninl0.get(v, 0) for v in adj.get(n, [])) for n in nodes}
            ninl2 = {n: sum(ninl1.get(v, 0) for v in adj.get(n, [])) for n in nodes}

            for n in nodes:
                risk_cache[n].append(ninl2[n])

        for n in nodes:
            if risk_cache[n]:
                avg = float(np.mean(risk_cache[n]))
                dq = risk_deques[n]
                dq.append(avg)
                if len(dq) > n_past:
                    dq.popleft()

        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def compute_average(self, risk_deques):
        return {n: sum(dq) / len(dq) for n, dq in risk_deques.items()}


class Ninl2RiskComputation(RiskComputation):
    """
    NINL-layer-2 risk using live degrees (honors removed edges).
    Layer-0: deg(u) + Σ_deg(v) over 1-hop neighbors
    Layer-1: Σ_layer0(v) over 1-hop neighbors
    Returned risk = layer-1
    """

    def __init__(self, normalize_risk_score: bool = False,
                 precomputed_max_risk: Optional[float] = None):
        super().__init__(normalize_risk_score, precomputed_max_risk)

    def initialize(self,
                   precomputed_degrees: Dict[int, Dict[int, int]],
                   adjacency_lists: List[Dict[int, Set[int]]],
                   maxlen: int = 5):
        self.adjacency_lists = adjacency_lists
        self.n_nodes = len(precomputed_degrees)
        nodes = list(precomputed_degrees.keys())

        adj0 = adjacency_lists[0]
        deg0 = {n: len(adj0.get(n, ())) for n in nodes}

        ninl0 = {n: deg0[n] + sum(deg0[v] for v in adj0.get(n, ())) for n in nodes}
        ninl1 = {n: sum(ninl0[v] for v in adj0.get(n, ())) for n in nodes}

        dq = {n: deque([ninl1[n]], maxlen=maxlen) for n in nodes}
        return self.normalize_risks_global(dq) if self.normalize_risk_score else dq

    def update(self, risk_deques, adjacency_lists, current_time,
               intervention_interval, removed_edges, n_past: int = 2):
        start_t = max(0, current_time - intervention_interval)
        nodes = range(self.n_nodes)
        cache: Dict[int, List[float]] = defaultdict(list)

        for t in range(start_t, current_time):
            ts = t % len(adjacency_lists)

            # adjusted live adjacency
            adj_live = {n: set(adjacency_lists[ts].get(n, ())) for n in nodes}
            for (u, v, ts_rm) in removed_edges:
                if ts_rm == ts:
                    adj_live[u].discard(v)
                    adj_live[v].discard(u)

            deg = {n: len(adj_live[n]) for n in nodes}

            ninl0 = {n: deg[n] + sum(deg[v] for v in adj_live[n]) for n in nodes}
            ninl1 = {n: sum(ninl0[v] for v in adj_live[n]) for n in nodes}

            for n in nodes:
                cache[n].append(ninl1[n])

        for n in nodes:
            if cache[n]:
                risk = float(np.mean(cache[n]))
                dq = risk_deques[n]
                dq.append(risk)
                if len(dq) > n_past:
                    dq.popleft()

        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def compute_average(self, risk_deques):
        return {n: sum(dq) / len(dq) for n, dq in risk_deques.items()}


# ──────────────────────────────────────────────────────────────────────────────
# ERM risk
# ──────────────────────────────────────────────────────────────────────────────

class ERMRiskComputation(RiskComputation):
    """
    ERM-based risk using 2-hop neighborhoods and entropic weights.
    Honors removed edges at each snapshot.
    """

    def __init__(self,
                 normalize_risk_score: bool = False,
                 precomputed_max_risk: Optional[float] = None):
        super().__init__(normalize_risk_score=normalize_risk_score,
                         precomputed_max_risk=precomputed_max_risk)

    def initialize(self,
                   precomputed_degrees: Dict[int, Dict[int, int]],
                   adjacency_lists: List[Dict[int, Set[int]]],
                   maxlen: int = 5):
        self.precomputed_degrees = precomputed_degrees
        self.adjacency_lists = adjacency_lists
        self.n_nodes = len(precomputed_degrees)
        nodes = list(precomputed_degrees.keys())

        adj0 = adjacency_lists[0]
        degree0 = {n: self.precomputed_degrees[n].get(0, len(adj0.get(n, set()))) for n in nodes}

        d1 = {n: sum(degree0[v] for v in adj0.get(n, set())) for n in nodes}
        d2 = {n: sum(d1[v] for v in adj0.get(n, set())) for n in nodes}

        E1, E2 = {}, {}
        for n in nodes:
            nbrs = adj0.get(n, set())
            if d1[n] > 0:
                E1[n] = -sum((degree0[v] / d1[n]) * math.log(degree0[v] / d1[n]) for v in nbrs)
            else:
                E1[n] = 0.0
            if d2[n] > 0:
                E2[n] = -sum((d1[v] / d2[n]) * math.log(d1[v] / d2[n]) for v in nbrs)
            else:
                E2[n] = 0.0

        maxE2 = max(E2.values()) if E2 else 0.0

        EC, SI = {}, {}
        for n in nodes:
            lam = (E2[n] / maxE2) if maxE2 > 0 else 0.0
            nbrs = adj0.get(n, set())
            EC[n] = sum(E1[v] + lam * E2[v] for v in nbrs)

        for n in nodes:
            SI[n] = sum(EC[v] for v in adj0.get(n, set()))

        risk_deques = {n: deque([SI[n]], maxlen=maxlen) for n in nodes}
        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def update(self,
               risk_deques: Dict[int, deque],
               adjacency_lists: List[Dict[int, Set[int]]],
               current_time: int,
               intervention_interval: int,
               removed_edges: List[Tuple[int, int, int]],
               n_past: int = 2):
        start_t = max(0, current_time - intervention_interval)
        nodes = list(self.precomputed_degrees.keys())
        cache: Dict[int, List[float]] = defaultdict(list)

        for t in range(start_t, current_time):
            ts = t % len(adjacency_lists)

            # adjusted live adjacency as sets
            adj_live: Dict[int, Set[int]] = {n: set(adjacency_lists[ts].get(n, set())) for n in nodes}
            for u, v, ts_rm in removed_edges:
                if ts_rm == ts:
                    adj_live[u].discard(v)
                    adj_live[v].discard(u)

            degree_t = {n: self.precomputed_degrees[n].get(ts, len(adj_live[n])) for n in nodes}

            d1, d2, E1, E2, EC, SI = {}, {}, {}, {}, {}, {}
            for n in nodes:
                nbrs = adj_live[n]
                d1[n] = sum(degree_t[v] for v in nbrs)
            for n in nodes:
                nbrs = adj_live[n]
                d2[n] = sum(d1[v] for v in nbrs)

            for n in nodes:
                nbrs = adj_live[n]
                if d1[n] > 0:
                    E1[n] = -sum((degree_t[v] / d1[n]) * math.log(degree_t[v] / d1[n]) for v in nbrs)
                else:
                    E1[n] = 0.0
                if d2[n] > 0:
                    E2[n] = -sum((d1[v] / d2[n]) * math.log(d1[v] / d2[n]) for v in nbrs)
                else:
                    E2[n] = 0.0

            maxE2 = max(E2.values()) if E2 else 0.0
            for n in nodes:
                lam = (E2[n] / maxE2) if maxE2 > 0 else 0.0
                nbrs = adj_live[n]
                EC[n] = sum(E1[v] + lam * E2[v] for v in nbrs)

            for n in nodes:
                SI[n] = sum(EC[v] for v in adj_live[n])

            # final ERM value per node for this snapshot
            for n in nodes:
                erm_val = sum(SI[v] for v in adj_live[n])
                cache[n].append(erm_val)

        for n in nodes:
            if cache[n]:
                avg = float(np.mean(cache[n]))
                dq = risk_deques[n]
                dq.append(avg)
                if len(dq) > n_past:
                    dq.popleft()

        return self.normalize_risks_global(risk_deques) if self.normalize_risk_score else risk_deques

    def compute_average(self, risk_deques: Dict[int, deque]) -> Dict[int, float]:
        return {n: sum(dq) / len(dq) for n, dq in risk_deques.items()}
