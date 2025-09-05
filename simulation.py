#!/usr/bin/env python3
"""
SIR + intervention runner.


Main entry points
-----------------
- run_sir_with_interventions(...): runs a single simulation on a temporal network.
- run_sir_for_all_nodes(...): orchestrates N runs (optionally with multiprocessing).

Returned data structure (exact keys & shapes)
--------------------------------------------
For a single simulation, the returned dict has:

    data = {
        # States over time
        'history': np.ndarray of shape (n_timesteps + 1, n_nodes),
            # SIR state per node at each time step (0=S, 1=I, 2=R). Includes t=0.

        # Edge-removal bookkeeping per intervention step
        'edge_removal_history': np.ndarray of shape (num_interventions,),
            # Number of edges removed at each intervention step.
        'removed_edges': list[list[tuple[int,int,int]]],
            # For each intervention step: list of edges removed as (u, v, t_index).
        'cumulative_removed_edges': np.ndarray of shape (num_interventions,),
            # Cumulative % of edges removed w.r.t. total edges in the *original* network.

        # Network size over time (per snapshot index)
        'edge_counts': np.ndarray of shape (n_timesteps,),
            # Remaining edges per snapshot after removals applied.
        'percentage_edges_per_snapshot_remaining': np.ndarray of shape (n_timesteps,),
            # 0–100, relative to original edges in that snapshot.

        # Risk / threshold / epidemiology per intervention step
        'high_risk_node_counts': np.ndarray of shape (num_interventions,),
        'high_risk_nodes': list[set[int]],
            # Set of node ids flagged as high risk at that step.
        'dynamic_thresholds': np.ndarray of shape (num_interventions,),
            # θ_t used at that intervention (0 if is_baseline_run).
        'avg_R0_change': np.ndarray of shape (num_interventions,),
            # Average ΔR0 reported by the intervention routine (if available).
        'P_t': np.ndarray of shape (num_interventions,),
            # Global infection potential used in threshold dynamics.
        'acceleration': np.ndarray of shape (num_interventions,),
            # Infection acceleration metric from intervention handler.
        'R0_values': np.ndarray of shape (num_interventions,),
            # Instantaneous R0 proxy at intervention times.
        'R0_average_values': np.ndarray of shape (num_interventions,),
            # Moving average of the R0 proxy.

        # Risk scores (sparse-filled by role)
        'risk_scores_over_time': np.ndarray of shape (num_interventions, n_nodes),
            # Average risk score per node at each intervention (0 if undefined).
        'high_risk_scores': np.ndarray of shape (num_interventions, n_nodes),
            # risk score per node but values are only set for nodes in high_risk_nodes.
        'non_high_risk_scores': np.ndarray of shape (num_interventions, n_nodes),
            # risk score per node but values are only set for nodes NOT in high_risk_nodes.

        # Per-node reduction accounting (provided by intervention.apply_intervention)
        'risk_reduction_data': {
            'risk_scores': np.ndarray of shape (num_interventions, n_nodes),
            'edges_reduced': np.ndarray of shape (num_interventions, n_nodes),
            'removal_fractions': np.ndarray of shape (num_interventions, n_nodes),
            'contribution_to_total_reduction': np.ndarray of shape (num_interventions, n_nodes),
        },

        # Timing
        'timing_info': {
            'intervention_times': np.ndarray of shape (num_interventions,),
            'total_time': float,
        },
    }

Notes
-----
- Edge tuples are stored as (min(u), max(v), t_index) consistently with membership checks.
- Other risk models (ninl2/3/4, erm) remain available, though the public results use 'degree'.
"""

import math
import time
import random
from collections import deque
from typing import List, Dict, Set, Tuple, Optional

import numpy as np

from sir_functions import (
    sir_simulation,
    compute_R0_from_adj_list,
    precompute_degrees,
    precompute_edge_lifespans,
)
from intervention import Intervention
from risk_computation import (
    DegreeBasedRiskComputation,
    Ninl4RiskComputation,
    Ninl3RiskComputation,
    Ninl2RiskComputation,
    ERMRiskComputation,
)


def run_sir_with_interventions(
    sim_id: int,
    adjacency_lists: List[Dict[int, Set[int]]],
    precomputed_degrees: Dict[int, Dict[int, int]],
    node_edge_lifespans: Dict[int, Dict[Tuple[int, int], List[int]]],
    beta: float,
    gamma: float,
    intervention: bool,
    removal_strategy: str,
    threshold: float,
    window: int,
    intervention_interval: int,
    n_timesteps: int,
    proportion_high_risk: float,
    original_edges_per_snapshot: List[int],
    original_network_total_edge_count: int,
    n_nodes: int,
    scaling_exponent: float,
    max_removal_fraction: Optional[float],
    min_removal_fraction: float,
    adjustment_step: float,
    r0_change_window_size: Optional[int],
    n_past_risk_values: Optional[int],
    risk_model: str,
    percentage_of_initial_infected: float,
    normalize_risk_score: bool,
    live_plot: bool = False,
    live_plot_callback=None,
    precomputed_max_risk: Optional[float] = None,
    rise_strength: float = 0.1,
    drop_strength: float = 1.0,
    accel_weight: float = 1.0,
    pt_weight: float = 1.0,
    rise_smoothing: float = 0.1,
    drop_smoothing: float = 0.7,
    use_dynamic_threshold: bool = True,
    fixed_threshold: Optional[float] = None,
    fixed_removal_fraction: Optional[float] = None,
    top_node_frac: Optional[float] = None,
    is_baseline_run: bool = False,
    compliance: float = 1.0,
) -> Dict[str, object]:
    """Run a single SIR simulation with optional interventions.

    Parameters mirror legacy code to preserve compatibility. See module docstring for returns.
    """
    np.random.seed(sim_id)
    random.seed(sim_id)

    start_time_total = time.time()
    states = np.zeros(n_nodes, dtype=np.uint8)  # 0=S, 1=I, 2=R

    # Infect a random set of nodes initially
    num_to_infect = int(math.ceil(percentage_of_initial_infected * n_nodes))
    random_nodes = np.random.choice(n_nodes, size=num_to_infect, replace=False)

    # Allocate result buffers
    num_history_steps = n_timesteps + 1
    num_intervention_steps = math.ceil(n_timesteps / intervention_interval)

    data: Dict[str, object] = {
        'history': np.zeros((num_history_steps, n_nodes), dtype=np.uint8),
        'edge_removal_history': np.zeros(num_intervention_steps, dtype=int),
        'edge_counts': np.zeros(n_timesteps, dtype=int),
        'percentage_edges_per_snapshot_remaining': np.zeros(n_timesteps, dtype=float),
        'high_risk_node_counts': np.zeros(num_intervention_steps, dtype=int),
        'R0_values': np.zeros(num_intervention_steps, dtype=float),
        'R0_average_values': np.zeros(num_intervention_steps, dtype=float),
        'cumulative_removed_edges': np.zeros(num_intervention_steps, dtype=float),

        'dynamic_thresholds': np.zeros(num_intervention_steps, dtype=float),
        'avg_R0_change': np.zeros(num_intervention_steps, dtype=float),
        'P_t': np.zeros(num_intervention_steps, dtype=float),
        'acceleration': np.zeros(num_intervention_steps, dtype=float),

        'high_risk_scores': np.zeros((num_intervention_steps, n_nodes), dtype=float),
        'non_high_risk_scores': np.zeros((num_intervention_steps, n_nodes), dtype=float),
        'high_risk_nodes': [set() for _ in range(num_intervention_steps)],
        'risk_scores_over_time': np.zeros((num_intervention_steps, n_nodes), dtype=float),

        'removed_edges': [],
        'risk_reduction_data': {
            'risk_scores': np.zeros((num_intervention_steps, n_nodes), dtype=float),
            'edges_reduced': np.zeros((num_intervention_steps, n_nodes), dtype=int),
            'removal_fractions': np.zeros((num_intervention_steps, n_nodes), dtype=float),
            'contribution_to_total_reduction': np.zeros((num_intervention_steps, n_nodes), dtype=float),
        },

        'timing_info': {
            'intervention_times': np.zeros(num_intervention_steps, dtype=float),
            'total_time': 0.0,
        },
    }

    history_index = 0
    data['history'][history_index, :] = states.copy()
    history_index += 1

    # Choose risk computer (keep support for other models, default to degree)
    if risk_model == "degree":
        risk_computer = DegreeBasedRiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)
    elif risk_model == "ninl2":
        risk_computer = Ninl2RiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)
    elif risk_model == "ninl3":
        risk_computer = Ninl3RiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)
    elif risk_model == "ninl4":
        risk_computer = Ninl4RiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)
    elif risk_model == "erm":
        risk_computer = ERMRiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)
    else:
        risk_computer = DegreeBasedRiskComputation(normalize_risk_score=normalize_risk_score, precomputed_max_risk=precomputed_max_risk)

    risk_deques = risk_computer.initialize(precomputed_degrees, adjacency_lists, maxlen=n_past_risk_values)

    # Intervention handler
    intervention_handler = Intervention(
        node_edge_lifespans=node_edge_lifespans,
        removal_strategy=removal_strategy,
        threshold=threshold,
        use_dynamic_threshold=use_dynamic_threshold,
        fixed_threshold=fixed_threshold,
        fixed_removal_fraction=fixed_removal_fraction,
        window=window,
        proportion_high_risk=proportion_high_risk,
        max_removal_fraction=max_removal_fraction,
        min_removal_fraction=min_removal_fraction,
        adjustment_step=adjustment_step,
        intervention_interval=intervention_interval,
        r0_change_window_size=r0_change_window_size,
        n_nodes=n_nodes,
        scaling_exponent=scaling_exponent,
    )

    removed_edges: Set[Tuple[int, int, int]] = set()
    total_removed_edges = 0
    R0_deque: deque = deque(maxlen=intervention_interval)
    removed_high_risk_nodes: Set[int] = set()

    intervention_index = 0

    # Main loop: advance in increments of intervention_interval
    for t in range(0, n_timesteps, intervention_interval):
        # Update risk computations up to the current time
        if t > 0:
            risk_deques = risk_computer.update(
                risk_deques=risk_deques,
                adjacency_lists=adjacency_lists,
                current_time=t,
                intervention_interval=intervention_interval,
                removed_edges=removed_edges,
                n_past=n_past_risk_values,
            )

        # Infection acceleration and global potential
        acceleration = intervention_handler.compute_infection_acceleration(data['history'], history_index, window=window)
        data['acceleration'][intervention_index] = acceleration
        data['P_t'][intervention_index] = intervention_handler.compute_global_infection_potential(states)

        # Dynamic threshold (or θ=0 baseline)
        if is_baseline_run:
            thr = 0.0
        else:
            thr = intervention_handler.compute_dynamic_threshold(
                acceleration,
                base_threshold=1.0,
                rise_strength=rise_strength,
                drop_strength=drop_strength,
                accel_weight=accel_weight,
                pt_weight=pt_weight,
                rise_smoothing=rise_smoothing,
                drop_smoothing=drop_smoothing,
            )
        data['dynamic_thresholds'][intervention_index] = thr

        # Identify high-risk nodes
        high_risk_nodes = intervention_handler.identify_high_risk_nodes(
            risk_deques, states, thr, fixed_threshold, threshold, top_node_frac
        )
        data['high_risk_nodes'][intervention_index] = high_risk_nodes
        non_high_risk_nodes = set(range(n_nodes)) - set(high_risk_nodes)

        # Current R0 proxy
        current_time_mod = t % len(adjacency_lists)
        R0_current = compute_R0_from_adj_list(
            adjacency_lists[current_time_mod],
            removed_edges,
            current_time_mod,
        )
        R0_deque.append(R0_current)
        R0_avg = np.mean(R0_deque) if R0_deque else 0.0

        # Apply intervention (skip at t=0 so we have initial risk history)
        if intervention and t > 0:
            start_intervention_time = time.time()

            # Average risk scores per node for this step
            risk_scores = risk_computer.compute_average(risk_deques)
            data['risk_scores_over_time'][intervention_index, :] = np.array(
                [risk_scores.get(node, 0.0) for node in range(n_nodes)]
            )

            # Build per-node removal fraction ("fixed depth" when fixed_removal_fraction is set)
            node_depth: Dict[int, float] = {}
            for node in high_risk_nodes:
                if fixed_removal_fraction is not None:
                    node_depth[node] = fixed_removal_fraction
                else:
                    node_depth[node] = intervention_handler.compute_removal_fraction(
                        risk_score=risk_scores[node],
                        P_t=intervention_handler.P_t,
                        scaling_exponent=scaling_exponent,
                        min_removal=min_removal_fraction,
                        max_removal=max_removal_fraction,
                    )

            result = intervention_handler.apply_intervention(
                current_time=t,
                adjacency_lists=adjacency_lists,
                high_risk_nodes=high_risk_nodes,
                R0_avg=R0_avg,
                risk_deques=risk_deques,
                states=states,
                intervention_interval=intervention_interval,
                R0_deque=R0_deque,
                risk_computer=risk_computer,
                n_nodes=n_nodes,
                removed_high_risk_nodes=removed_high_risk_nodes,
                removed_edges=removed_edges,
                node_removal_fractions=node_depth,
                compliance=compliance,
            )
            data['timing_info']['intervention_times'][intervention_index] = time.time() - start_intervention_time

            # Unpack results from intervention
            if isinstance(result, tuple) and len(result) > 4:
                (
                    edge_removal_count,
                    cur_removed_edges,
                    removed_edges,
                    removed_high_risk_nodes,
                    current_high_risk_nodes,
                    risk_reduction_data,
                    avg_R0_change,
                    P_t,
                    adjacency_lists,
                ) = result

                total_removed_edges += len(cur_removed_edges)
                data['removed_edges'].append(list(cur_removed_edges))
                data['edge_removal_history'][intervention_index] = len(cur_removed_edges)
                data['avg_R0_change'][intervention_index] = avg_R0_change
                data['P_t'][intervention_index] = P_t

                # Per-node reduction accounting
                data['risk_reduction_data']['risk_scores'][intervention_index, :] = risk_reduction_data['risk_scores']
                data['risk_reduction_data']['edges_reduced'][intervention_index, :] = risk_reduction_data['edges_reduced']
                data['risk_reduction_data']['removal_fractions'][intervention_index, :] = risk_reduction_data['removal_fractions']
                data['risk_reduction_data']['contribution_to_total_reduction'][intervention_index, :] = risk_reduction_data['contribution_to_total_reduction']

                for node in high_risk_nodes:
                    data['high_risk_scores'][intervention_index, node] = risk_reduction_data['risk_scores'][node]
                for node in non_high_risk_nodes:
                    data['non_high_risk_scores'][intervention_index, node] = risk_reduction_data['risk_scores'][node]
            else:
                # No edges removed (or different return shape) → store minimal data
                data['removed_edges'].append([])
                data['edge_removal_history'][intervention_index] = 0
        else:
            data['removed_edges'].append([])
            data['edge_removal_history'][intervention_index] = 0

        # Summary stats for this intervention step
        data['high_risk_node_counts'][intervention_index] = len(high_risk_nodes)
        data['R0_values'][intervention_index] = R0_current
        data['R0_average_values'][intervention_index] = R0_avg
        data['cumulative_removed_edges'][intervention_index] = (
            (total_removed_edges / original_network_total_edge_count) * 100.0
            if original_network_total_edge_count > 0
            else 0.0
        )

        # Simulate SIR for the next `intervention_interval` steps (or fewer near the end)
        steps_to_simulate = min(intervention_interval, n_timesteps - t)
        sir_gen = sir_simulation(
            initial_infected_nodes=random_nodes,
            adjacency_lists=adjacency_lists,
            start_t=t,
            beta=beta,
            gamma=gamma,
            n_timesteps=steps_to_simulate,
            states=states,
            removed_edges=removed_edges,
        )

        for new_states in sir_gen:
            data['history'][history_index, :] = new_states.copy()
            history_index += 1
            states[:] = new_states  # Update the outer states array in place

            current_step = history_index - 1
            # Map step to snapshot index within the temporal network window
            current_snapshot_index = min(t + (history_index - 1) % intervention_interval, len(adjacency_lists) - 1)

            # Edge count & percentage for this snapshot
            adj_list = adjacency_lists[current_snapshot_index]
            edge_count = sum(
                1
                for node, neighbors in adj_list.items()
                for neighbor in neighbors
                if (min(node, neighbor), max(node, neighbor), current_snapshot_index) not in removed_edges
            ) // 2
            data['edge_counts'][current_snapshot_index] = edge_count

            orig = original_edges_per_snapshot[current_snapshot_index]
            data['percentage_edges_per_snapshot_remaining'][current_snapshot_index] = (
                (edge_count / orig) * 100.0 if orig > 0 else 100.0
            )

            # Live plotting tick (every 240 steps by default)
            if live_plot and (live_plot_callback is not None) and (current_step % 240 == 0):
                live_plot_callback(
                    data=data,
                    states=new_states,
                    removed_edges=removed_edges,
                    adjacency_lists=adjacency_lists,
                    current_step=current_step,
                    intervention_step=intervention_index,
                    risk_deques=risk_deques,
                    intervention=intervention,
                )

        intervention_index += 1

    # After main loop, finalize per-snapshot edge counts (idempotent if already set)
    for i in range(len(adjacency_lists)):
        edge_count = sum(
            1
            for node, neighbors in adjacency_lists[i].items()
            for neighbor in neighbors
            if (min(node, neighbor), max(node, neighbor), i) not in removed_edges
        ) // 2
        data['edge_counts'][i] = edge_count
        orig = original_edges_per_snapshot[i]
        data['percentage_edges_per_snapshot_remaining'][i] = (
            (edge_count / orig) * 100.0 if orig > 0 else 100.0
        )

    data['timing_info']['total_time'] = time.time() - start_time_total

    if sim_id % 1 == 0:
        print(f"Simulation {sim_id} done. Time = {data['timing_info']['total_time']:.2f}s")

    return data


# Wrapper for multiprocessing

def worker_sim(args):
    return run_sir_with_interventions(*args)


def run_sir_for_all_nodes(
    temporal_network,
    n_simulations,
    beta,
    gamma,
    n_nodes,
    scaling_exponent,
    intervention=False,
    removal_strategy="full_duration",
    threshold=0.7,
    window=50,
    intervention_interval=50,
    n_timesteps=50,
    proportion_high_risk=0.1,
    n_processes=1,
    use_multiprocessing=False,
    max_removal_fraction=0.3,
    min_removal_fraction=0.0,
    adjustment_step=0.05,
    r0_change_window_size=None,
    n_past_risk_values=None,
    risk_model="degree",
    percentage_of_initial_infected=0.05,
    normalize_risk_score=False,
    live_plot=False,
    live_plot_callback=None,
    precomputed_max_risk=None,
    rise_strength=0.1,
    drop_strength=1.0,
    accel_weight=1.0,
    pt_weight=1.0,
    rise_smoothing=0.1,
    drop_smoothing=0.7,
    use_dynamic_threshold=True,
    fixed_threshold=None,
    fixed_removal_fraction=None,
    top_node_frac=None,
    is_baseline_run=False,
    compliance=1.0,
):
    """
    Orchestrate multiple SIR simulations on a given temporal network.

    Returns
    -------
    dict: results[simulation_id] = single-simulation dict (see module docstring).
    """
    import gc
    from multiprocessing import Pool

    # Convert each snapshot Graph to adjacency list
    adjacency_lists = []
    for G in temporal_network:
        adj_list = {node: set(G.neighbors(node)) for node in G.nodes()}
        adjacency_lists.append(adj_list)

    # Precomputations
    precomputed_degrees = precompute_degrees(adjacency_lists)
    edge_lifespans, node_edge_lifespans = precompute_edge_lifespans(temporal_network)

    original_edges_per_snapshot = [G.number_of_edges() for G in temporal_network]
    original_network_total_edge_count = sum(original_edges_per_snapshot)

    # Build tasks for each simulation
    def build_task(sim_id: int):
        return (
            sim_id,
            adjacency_lists,
            precomputed_degrees,
            node_edge_lifespans,
            beta,
            gamma,
            intervention,
            removal_strategy,
            threshold,
            window,
            intervention_interval,
            n_timesteps,
            proportion_high_risk,
            original_edges_per_snapshot,
            original_network_total_edge_count,
            n_nodes,
            scaling_exponent,
            max_removal_fraction,
            min_removal_fraction,
            adjustment_step,
            r0_change_window_size,
            n_past_risk_values,
            risk_model,
            percentage_of_initial_infected,
            normalize_risk_score,
            live_plot,
            live_plot_callback,
            precomputed_max_risk,
            rise_strength,
            drop_strength,
            accel_weight,
            pt_weight,
            rise_smoothing,
            drop_smoothing,
            use_dynamic_threshold,
            fixed_threshold,
            fixed_removal_fraction,
            top_node_frac,
            is_baseline_run,
            compliance,
        )

    tasks = [build_task(sim_id) for sim_id in range(n_simulations)]
    results = {}

    if use_multiprocessing and n_processes > 1:
        with Pool(n_processes) as pool:
            results_list = pool.map(worker_sim, tasks)
        for idx, sim_result in enumerate(results_list):
            results[idx] = sim_result
    else:
        for idx, task in enumerate(tasks):
            results[idx] = worker_sim(task)

    gc.collect()
    return results
