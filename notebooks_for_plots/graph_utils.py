#!/usr/bin/env python3
"""
Graph utilities for NPCT simulations.

Place your files like this (or set env vars listed below):
  - DTU CSV: `./data/bt_symmetric.csv`  (or env `DTU_CSV`)
  - ABM CSV: `./data/micro_abm_contacts.csv`      (or env `ABM_CONTACTS`)
  - ABM30 CSV: `./data/micro_abm_contacts30.csv`  (or env `ABM30_CONTACTS`)
  - Workplace: `./data/workplace.dat` with columns [Timestamp, PersonId1, PersonId2]

Supported `load_network(name)` values: 'DTU', 'abm', 'abm30', 'workplace'.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple, Dict, Any, Set, Optional

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Global timing constants for (optional) aggregation/cropping
# ──────────────────────────────────────────────────────────────────────────────
HOUR: int = 3600
SNAPSHOTS_PER_DAY: int = 24
MAX_DAYS_DEFAULT: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# ABM loader (unchanged logic; now path is provided by caller or env)
# ──────────────────────────────────────────────────────────────────────────────

def load_abm_network(
    file_path: str,
    remap: bool = True,
    node_attributes: Optional[Dict[int, Dict[str, int]]] = None,
    required_ageGroup: Optional[str] = None,
) -> List[nx.Graph]:
    """
    Load an ABM temporal network CSV into a list of graphs (one per Hour).
    CSV is expected to have columns:
      Hour, PersonId1, PersonId2, Intensity, LocationId, LocationType
    """
    df = pd.read_csv(
        file_path,
        comment="#",
        names=[
            "Hour",
            "PersonId1",
            "PersonId2",
            "Intensity",
            "LocationId",
            "LocationType",
        ],
    )

    # Ensure numeric types
    for c in ("Hour", "PersonId1", "PersonId2", "Intensity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)

    all_nodes = set(df["PersonId1"]).union(set(df["PersonId2"]))
    temporal_network: List[nx.Graph] = []
    hours = sorted(df["Hour"].unique())

    for hour in tqdm(hours, desc="Loading ABM network"):
        group = df[df["Hour"] == hour]
        G = nx.from_pandas_edgelist(
            group,
            "PersonId1",
            "PersonId2",
            edge_attr=["Intensity", "LocationId", "LocationType"],
        )

        # Ensure every node appears in this snapshot
        G.add_nodes_from(all_nodes)

        # Copy 'Intensity' into 'weight'
        for u, v, data in G.edges(data=True):
            data["weight"] = data["Intensity"]

        temporal_network.append(G)

    if remap:
        temporal_network = remap_nodes(
            temporal_network,
            node_attributes=node_attributes,
            required_ageGroup=required_ageGroup,
        )
    return temporal_network


# ──────────────────────────────────────────────────────────────────────────────
# Node remapping & small utilities
# ──────────────────────────────────────────────────────────────────────────────

def remap_nodes(
    temporal_network: List[nx.Graph],
    node_attributes: Optional[Dict[int, Dict[str, int]]] = None,
    required_ageGroup: Optional[str] = None,
) -> List[nx.Graph]:
    """Remap nodes consistently across snapshots and optionally filter by attribute."""
    global_node_mapping: Dict[Any, int] = {}
    next_node_id = 0

    # Build global mapping
    for snapshot in temporal_network:
        for node in snapshot.nodes():
            if node not in global_node_mapping:
                global_node_mapping[node] = next_node_id
                next_node_id += 1

    # Filter set (optional)
    if node_attributes:
        valid_nodes = set(global_node_mapping.keys())
        if required_ageGroup:
            valid_nodes = {
                node
                for node in valid_nodes
                if node in node_attributes
                and node_attributes[node].get("age_group_name", "") == required_ageGroup
            }
    else:
        valid_nodes = set(global_node_mapping.keys())

    remapped: List[nx.Graph] = []
    for snapshot in temporal_network:
        g_new = nx.Graph()
        # Add all valid nodes (even if isolated)
        g_new.add_nodes_from(global_node_mapping[n] for n in valid_nodes)
        # Add edges if both endpoints valid
        for u, v, data in snapshot.edges(data=True):
            if u in valid_nodes and v in valid_nodes:
                uu = global_node_mapping[u]
                vv = global_node_mapping[v]
                if uu != vv:
                    g_new.add_edge(uu, vv, **data)
        remapped.append(g_new)
    return remapped


def get_individuals_from_graph(temporal_network: List[nx.Graph]) -> List[int]:
    """Sorted list of all unique node IDs across the temporal network."""
    unique = set()
    for G in temporal_network:
        unique.update(G.nodes())
    return sorted(int(n) for n in unique)


def get_current_network(
    t: int,
    temporal_network: List[nx.Graph],
    removed_edges_per_snapshot: Dict[int, Set[Tuple[int, int]]],
) -> nx.Graph:
    """Return a copy of snapshot t with specific edges removed (utility)."""
    G = temporal_network[t].copy()
    G.remove_edges_from(removed_edges_per_snapshot.get(t, set()))
    return G


# ──────────────────────────────────────────────────────────────────────────────
# DTU loader (5-min → hourly, cropped to N days) + helpers for CSV-based graphs
# ──────────────────────────────────────────────────────────────────────────────

def load_df(
    file_name: str,
    n_individuals: Optional[int] = None,
    n_row: Optional[int] = None,
    seed: Optional[float] = None,
) -> pd.DataFrame:
    """Load a DTU-style CSV (user_a, user_b, rssi, # timestamp)."""
    df = pd.read_csv(file_name)

    # Remove invalid rows if present
    if "user_b" in df.columns:
        df = df.drop(df[(df.user_b == -1)].index)
        df = df.drop(df[(df.user_b == -2)].index)

    if seed is not None:
        random.seed(seed)

    if n_row is not None:
        df = df.head(n_row)

    if n_individuals is not None:
        df = remove_individuals(df, n_individuals)

    return df


def remove_individuals(df: pd.DataFrame, n_individuals: int) -> pd.DataFrame:
    """Keep only interactions among a random subset of n_individuals."""
    unique_a = df["user_a"].unique()
    unique_b = df["user_b"].unique()
    all_ids = np.unique(np.concatenate([unique_a, unique_b]))
    chosen = random.choices(all_ids, k=n_individuals)
    return df.loc[df["user_a"].isin(chosen) & df["user_b"].isin(chosen)]


def get_array_of_contacts(df: pd.DataFrame, temporal_gap: float, time_col: str) -> List[pd.DataFrame]:
    """Split interactions into windows of size `temporal_gap` using `time_col`."""
    static_contacts: List[pd.DataFrame] = []
    max_time = df[time_col].max()
    n_snapshots = int(max_time / temporal_gap) if max_time and temporal_gap else 0

    for i in range(n_snapshots):
        lower = i * temporal_gap
        upper = (i + 1) * temporal_gap
        if i == n_snapshots - 1:
            tmp = df.loc[df[time_col] >= lower]
        else:
            tmp = df.loc[(df[time_col] >= lower) & (df[time_col] < upper)]
        static_contacts.append(tmp.copy())
    return static_contacts


def build_graphs(static_contacts: List[pd.DataFrame], temporal_gap: float) -> List[nx.Graph]:
    """Build one graph per snapshot from DTU-style frames (user_a, user_b, rssi)."""
    graphs: List[nx.Graph] = []
    for frame in static_contacts:
        subset = frame[["user_a", "user_b", "rssi"]]
        G = nx.from_pandas_edgelist(subset, "user_a", "user_b", edge_attr="rssi")
        graphs.append(G)

    if not graphs:
        return graphs

    # Init first snapshot durations
    for e in graphs[0].edges():
        graphs[0].edges()[e]["duration"] = temporal_gap

    # Accumulate duration & mean RSSI over time
    for i in range(len(graphs) - 1):
        g0 = graphs[i]
        g1 = graphs[i + 1]
        for u, v in g1.edges():
            if g0.has_edge(u, v):
                old_rssi = g0[u][v]["rssi"]
                old_dur = g0[u][v].get("duration", 0.0)
                g1[u][v]["duration"] = old_dur + temporal_gap
                g1[u][v]["rssi"] = np.mean([g1[u][v]["rssi"], old_rssi])
            else:
                g1[u][v]["duration"] = temporal_gap
    return graphs


def get_graph_from_csv(
    csv_file: str,
    temporal_gap: float,
    n_individuals: Optional[int] = None,
    n_row: Optional[int] = None,
) -> List[nx.Graph]:
    """Load CSV → window into `temporal_gap` → build snapshot graphs."""
    df = load_df(csv_file, n_individuals=n_individuals, n_row=n_row)
    contacts_per_window = get_array_of_contacts(df, temporal_gap, time_col="# timestamp")
    return build_graphs(contacts_per_window, temporal_gap)


def get_DTU_graph(
    temporal_gap: float,
    n_individuals: Optional[int] = None,
    n_row: Optional[int] = None,
    csv_file: Optional[str] = None,
) -> List[nx.Graph]:
    """
    Load DTU temporal graph snapshots. Defaults to `DTU_CSV` or `./data/bt_symmetric.csv`.
    """
    if csv_file is None:
        csv_file = os.environ.get("DTU_CSV", os.path.join("data", "bt_symmetric.csv"))
    print(f"Loading DTU graph from {csv_file} with temporal_gap={temporal_gap}")
    return get_graph_from_csv(csv_file, temporal_gap, n_individuals=n_individuals, n_row=n_row)


def aggregate_temporal_network(net: List[nx.Graph], win: int = 1) -> List[nx.Graph]:
    """Merge each consecutive `win` snapshots (edge weight = #observations in the block)."""
    if win <= 1:
        return net
    aggregated: List[nx.Graph] = []
    for start in range(0, len(net), win):
        merged = nx.Graph()
        block = net[start : start + win]
        for g in block:
            for u, v, d in g.edges(data=True):
                w_old = merged[u][v]["weight"] if merged.has_edge(u, v) else 0
                merged.add_edge(u, v, weight=w_old + d.get("weight", 1.0))
        # retain isolated nodes
        for g in block:
            merged.add_nodes_from(g.nodes())
        aggregated.append(merged)
    return aggregated


def limit_to_days(net: List[nx.Graph], max_days: int = MAX_DAYS_DEFAULT, snaps_per_day: int = SNAPSHOTS_PER_DAY) -> List[nx.Graph]:
    """Keep at most `max_days * snaps_per_day` snapshots."""
    return net if max_days is None else net[: max_days * snaps_per_day]


def get_DTU_graph_hourly(
    csv_file: Optional[str] = None,
    burn_in_days: int = 2,
    max_days: int = MAX_DAYS_DEFAULT,
) -> List[nx.Graph]:
    """DTU 5-min raw → hourly snapshots; drop first `burn_in_days`; limit to `max_days`."""
    raw_5m = get_DTU_graph(temporal_gap=300, csv_file=csv_file)
    skip = burn_in_days * 288  # 288×5min = 1 day
    raw_5m = raw_5m[skip:]
    hourly = aggregate_temporal_network(raw_5m, win=12)  # 12×5min = 1 hour
    hourly = remap_nodes(hourly)
    return limit_to_days(hourly, max_days=max_days)


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point used by the batch script
# ──────────────────────────────────────────────────────────────────────────────
def load_network(name: str) -> List[nx.Graph]:
    name = name.lower()
    if name == "dtu":
        # Let get_DTU_graph_hourly -> get_DTU_graph read DTU_CSV or default
        return get_DTU_graph_hourly(csv_file=None)
    if name == "abm":
        return load_abm_network(os.environ.get("ABM_CONTACTS", os.path.join("data", "micro_abm_contacts.csv")))
    if name == "abm30":
        return load_abm_network(os.environ.get("ABM30_CONTACTS", os.path.join("data", "micro_abm_contacts30.csv")))
    if name == "workplace":
        file_path = os.environ.get("WORKPLACE_DAT", os.path.join("data", "workplace.dat"))
        return load_generic_network(
            file_path=file_path,
            time_window=HOUR,
            column_names=["Timestamp", "PersonId1", "PersonId2"],
            header=None,
            remap=True,
            time_col="Timestamp",
            aggregate_hourly=False,
            max_days=MAX_DAYS_DEFAULT,
        )
    raise ValueError(f"Unsupported network name: {name}")



# ──────────────────────────────────────────────────────────────────────────────
# Generic CSV-to-temporal loader used by 'workplace' 
# ──────────────────────────────────────────────────────────────────────────────

def load_generic_network(
    file_path: str,
    time_window: int,
    column_names: list,
    header: Optional[int] = None,
    remap: bool = True,
    node_attributes: Optional[Dict[int, Dict[str, int]]] = None,
    *,
    time_col: str = "Timestamp",
    aggregate_hourly: bool = True,
    max_days: int = MAX_DAYS_DEFAULT,
) -> List[nx.Graph]:
    """
    Generic temporal loader: read CSV, window by `time_window` seconds, ensure integer windows,
    (optionally) remap/filter nodes, (optionally) aggregate to hourly, and crop to `max_days`.
    """
    df = pd.read_csv(
        file_path,
        header=header,
        sep=None,  # auto-detect
        engine="python",
        names=column_names,
    )
    print(f"[LOAD] {file_path} – Shape: {df.shape}, Columns: {df.columns.tolist()}")

    if time_col not in df.columns:
        raise KeyError(f"{time_col} not in columns of {file_path}")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df["PersonId1"] = pd.to_numeric(df["PersonId1"], errors="coerce")
    df["PersonId2"] = pd.to_numeric(df["PersonId2"], errors="coerce")

    before = df.shape[0]
    df.dropna(subset=[time_col, "PersonId1", "PersonId2"], inplace=True)
    after = df.shape[0]
    if after == 0:
        print(f"WARNING: empty dataframe after dropna for {file_path}")
        return []
    if after < before:
        print(f"[CLEAN] dropped {before - after} invalid rows")

    # Window indices starting from 0
    df_min = df[time_col].min()
    df["Window"] = ((df[time_col] - df_min) // time_window).astype(int)

    start_w = int(df["Window"].min())
    end_w = int(df["Window"].max()) + 1
    time_windows = range(start_w, end_w)

    all_nodes = set(df["PersonId1"]).union(df["PersonId2"])
    temporal_net: List[nx.Graph] = []

    for w in tqdm(time_windows, desc=f"Loading {os.path.basename(file_path)}"):
        snap_df = df[df["Window"] == w]
        G = nx.from_pandas_edgelist(snap_df, "PersonId1", "PersonId2")
        G.add_nodes_from(all_nodes)
        # default unit weight per observation
        nx.set_edge_attributes(G, {e: 1.0 for e in G.edges()}, name="weight")
        temporal_net.append(G)

    if remap:
        temporal_net = remap_nodes(temporal_net, node_attributes=node_attributes)

    if aggregate_hourly and time_window != HOUR:
        win = max(1, int(round(HOUR / time_window)))
        temporal_net = aggregate_temporal_network(temporal_net, win=win)

    snaps_per_day = int(round(86400 / time_window))
    return limit_to_days(temporal_net, max_days=max_days, snaps_per_day=snaps_per_day)
