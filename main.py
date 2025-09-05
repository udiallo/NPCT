"""
Minimal, reproducible runner for our NPCT simulations.

Scope of this cleaned version (matches the paper results):
- Risk model: **degree** only.
- Intervention type: dynamic threshold (sensitivity λ) with **fixed removal fraction** ϕ.
- Baselines: (a) no intervention; (b) θ = 0 ("no-threshold") baseline that always removes edges
  for nodes above the (trivially lowest) cutoff at each step.
- Networks: DTU, abm, abm30, workplace (office).
"""

from __future__ import annotations

import os
import time
import cProfile
import pstats
from itertools import product

# Local imports
from graph_utils import load_abm_network, get_individuals_from_graph, load_network
from data_utils import save_data, save_parameters
from simulation import run_sir_for_all_nodes
from plotting_utils import live_plot_callback
from sir_functions import (
    precompute_degrees,
    compute_max_degree_risk,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Base parameters (restricted to what's used in the paper) ──────────────────
# ──────────────────────────────────────────────────────────────────────────────

BASE_PARAMS = {
    # Data / SIR
    "network_name": "workplace",  # one of: DTU | abm | abm30 | workplace
    "n_simulations": 200,
    "beta": None,  # set from network below
    "gamma": None, # set from network below
    "percentage_of_initial_infected": 0.05,

    # Intervention core
    "intervention_interval": 24,   # hours (∆t)
    "use_dynamic_threshold": True, # dynamic θ only
    "fixed_threshold": False,      # keep explicit for param files
    "threshold": 0.5,              # unused with dynamic θ but kept for completeness
    "window": 24,                  # look-back for acceleration / risk
    "risk_model": "degree",       # locked-in for this cleaned script

    # Removal fractions (ϕ)
    "min_removal_fraction": 0.0,
    "max_removal_fraction": None,  # not used in fixed-ϕ runs
    "fixed_removal_fraction": None,# set per run from sweep below

    # Sensitivity λ (named "drop_strength" in legacy code/plots)
    "drop_strength": None,         # set per run
    "drop_smoothing": 1.0,         # kept at 1.0 (paper default here)

    # Misc (kept for reproducibility/compatibility)
    "normalize_risk_score": True,
    "proportion_high_risk": 1.0,
    "adjustment_step": 0.1,
    "n_past_risk_values": 1,
    "r0_change_window_size": 24,
    "scaling_exponent": 0.2,
    "rise_strength": 1.0,
    "accel_weight": 1.0,
    "pt_weight": 1.0,
    "rise_smoothing": 1.0,
    "compliance": 1.0,

    # Baseline flags
    "is_baseline_run": False,

    # Execution
    "n_processes": 30,
    "use_multiprocessing": True,
    "live_plot": False,
}

# Set (β, γ) defaults by network (kept from original for reproducibility)
if BASE_PARAMS["network_name"] == "DTU":
    BASE_PARAMS["beta"] = 0.04
    BASE_PARAMS["gamma"] = 0.005
elif BASE_PARAMS["network_name"] == "abm":
    BASE_PARAMS["beta"] = 0.01
    BASE_PARAMS["gamma"] = 0.005
elif BASE_PARAMS["network_name"] == "workplace":
    BASE_PARAMS["beta"] = 0.3
    BASE_PARAMS["gamma"] = 0.005
elif BASE_PARAMS["network_name"] == "abm30":
    BASE_PARAMS["beta"] = 0.005
    BASE_PARAMS["gamma"] = 0.002

# ──────────────────────────────────────────────────────────────────────────────
# 2) Sweep settings (restricted to dynThresh + fixed ϕ) ────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

INTERVENTION_INTERVALS = [24]

# Sensitivity λ (legacy var name: drop_strength → "ds" in filenames/plots)
DROP_STRENGTHS = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
DROP_SMOOTHING = [1.0]

# Fixed removal fractions ϕ
FIXED_REMOVAL_FRACS = [0.10, 0.2, 0.25, 0.3, 0.4, 0.50, 0.6, 0.75, 1.00] # for abm30, only [0.10, 0.25, 0.50, 1.00] is needed

# Compliance kept constant here 
COMPLIANCES = [1.0] # for paper results [0.3, 0.6, 0.9, 1.0] was used

# Only the used variant: dynamic threshold + fixed removal fraction
VARIANTS = [
    {
        "label": "dynThresh_FRem",
        "use_dynamic_threshold": True,
        "fixed_threshold": False,
        "drop_strengths": DROP_STRENGTHS,
        "drop_smoothing": DROP_SMOOTHING,
        "max_removal_fracs": [None],
        "fixed_removal_fracs": FIXED_REMOVAL_FRACS,
        "top_node_fracs": [None],  # not used in this paper version
    }
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers ─────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fmt(x):
    return str(x).replace('.', 'p')


def make_suffix(p, ii, ds, mrf, frem, tnf):
    def f_or_na(name, val):
        return f"{name}{('NA' if val is None else f'{int(val*100):02d}')}"

    parts = [
        f"ii{ii}",
        f_or_na('ds', ds),
        f_or_na('mrf', mrf),
        f_or_na('frem', frem),
        f_or_na('tnf', tnf),
        f"ibl{int(p['is_baseline_run'])}",
        f"c{int(p['compliance']*100):02d}",
        f"ft{p['fixed_threshold']}",
        f"nprv{p['n_past_risk_values']}",
        f"rm{p['risk_model']}",
        f"w{fmt(p['window'])}",
        f"aw{fmt(p['accel_weight'])}",
        f"pt{fmt(p['pt_weight'])}",
        f"rs{fmt(p['rise_smoothing'])}",
        f"dsp{fmt(p['drop_smoothing'])}",
        f"b{fmt(p['beta'])}",
        f"g{fmt(p['gamma'])}",
    ]
    return "_".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Network prep (once) ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def _abm_path(name: str) -> str:
    # Prefer env vars; else relative ./data/
    defaults = {
        "abm": os.environ.get("ABM_CONTACTS", os.path.join("data", "micro_abm_contacts.csv")),
        "abm30": os.environ.get("ABM30_CONTACTS", os.path.join("data", "micro_abm_contacts30.csv")),
    }
    return defaults[name]


def build_network_and_globals(params):
    """Load temporal network, precompute degrees and max degree risk."""
    if params["network_name"] == "abm":
        temporal = load_abm_network(
            file_path=_abm_path("abm"),
            remap=True,
            node_attributes=None,
            required_ageGroup=None,
        )
    elif params["network_name"] == "abm30":
        temporal = load_abm_network(
            file_path=_abm_path("abm30"),
            remap=True,
            node_attributes=None,
            required_ageGroup=None,
        )
    else:
        temporal = load_network(params["network_name"])  # DTU, workplace, etc.

    n_nodes = len(get_individuals_from_graph(temporal))
    adj_lists = [{n: set(G.neighbors(n)) for n in G.nodes()} for G in temporal]

    precomp_deg = precompute_degrees(adj_lists)
    max_risk = compute_max_degree_risk(temporal, precomp_deg)

    return temporal, n_nodes, max_risk


# ──────────────────────────────────────────────────────────────────────────────
# 4) Batch driver ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def run_batch():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(script_dir, f"results_{BASE_PARAMS['network_name']}")
    os.makedirs(results_dir, exist_ok=True)

    # Load network + globals once
    temporal, n_nodes, max_risk = build_network_and_globals(BASE_PARAMS)
    n_timesteps = len(temporal)

    # Save base params snapshot
    save_parameters(
        results_dir,
        f"sim_params_{BASE_PARAMS['network_name']}_BASE.json",
        BASE_PARAMS,
    )

    # ── (A) Baselines: no intervention ───────────────────────────────────────
    baseline_total = (
        len(INTERVENTION_INTERVALS)
        * len(DROP_STRENGTHS)
        * len(DROP_SMOOTHING)
    )
    bcnt = 0

    for intervention_interval in INTERVENTION_INTERVALS:
        for ds in DROP_STRENGTHS:
            for dsmoothing in DROP_SMOOTHING:
                bcnt += 1
                print(f"\n[Baseline {bcnt}/{baseline_total}] no-intervention (ds={ds}, ii={intervention_interval}) …")

                p = BASE_PARAMS.copy()
                p.update(
                    {
                        "intervention": False,
                        "removal_strategy": "without",
                        "intervention_interval": intervention_interval,
                        "drop_strength": ds,
                        "drop_smoothing": dsmoothing,
                        "max_removal_fraction": None,
                        "fixed_removal_fraction": None,
                        "top_node_frac": None,
                    }
                )

                suffix = make_suffix(
                    p,
                    ii=intervention_interval,
                    ds=ds,
                    mrf=p["max_removal_fraction"],
                    frem=p["fixed_removal_fraction"],
                    tnf=p.get("top_node_frac"),
                )

                # Save params JSON
                params_fname = f"params_without_{suffix}.json"
                save_parameters(results_dir, params_fname, p)

                baseline = run_sir_for_all_nodes(
                    temporal_network=temporal,
                    n_simulations=p["n_simulations"],
                    beta=p["beta"],
                    gamma=p["gamma"],
                    n_nodes=n_nodes,
                    scaling_exponent=p["scaling_exponent"],
                    intervention=p["intervention"],
                    removal_strategy=p["removal_strategy"],
                    threshold=p["threshold"],
                    window=p["window"],
                    intervention_interval=p["intervention_interval"],
                    n_timesteps=n_timesteps,
                    proportion_high_risk=p["proportion_high_risk"],
                    n_processes=p["n_processes"],
                    use_multiprocessing=p["use_multiprocessing"],
                    max_removal_fraction=p["max_removal_fraction"],
                    min_removal_fraction=p["min_removal_fraction"],
                    adjustment_step=p["adjustment_step"],
                    r0_change_window_size=p["r0_change_window_size"],
                    n_past_risk_values=p["n_past_risk_values"],
                    risk_model=p["risk_model"],
                    percentage_of_initial_infected=p["percentage_of_initial_infected"],
                    normalize_risk_score=p["normalize_risk_score"],
                    live_plot=p["live_plot"],
                    live_plot_callback=live_plot_callback,
                    precomputed_max_risk=max_risk,
                    rise_strength=p["rise_strength"],
                    drop_strength=p["drop_strength"],
                    accel_weight=p["accel_weight"],
                    pt_weight=p["pt_weight"],
                    rise_smoothing=p["rise_smoothing"],
                    drop_smoothing=p["drop_smoothing"],
                    use_dynamic_threshold=p["use_dynamic_threshold"],
                    fixed_threshold=p["fixed_threshold"],
                    fixed_removal_fraction=p["fixed_removal_fraction"],
                    top_node_frac=p.get("top_node_frac"),
                    is_baseline_run=p["is_baseline_run"],
                )

                # Save ZIP
                zip_fname = f"results_without_{suffix}_BASELINE.zip"
                save_data(
                    os.path.join(results_dir, zip_fname),
                    {"without": list(baseline.values())},
                    f"without_{suffix}_BASELINE",
                )
                print(f"    ↳ baseline saved → {zip_fname}")

    # ── (B) Intervention runs: dynThresh + fixed ϕ + θ=0 baseline ─────────────
    inter_total = sum(
        len(v["drop_strengths"]) *
        len(v["max_removal_fracs"]) *
        len(v["fixed_removal_fracs"]) *
        len(v["top_node_fracs"]) *
        len(v["drop_smoothing"]) *
        len(COMPLIANCES)
        for v in VARIANTS
    )

    cnt = 0
    for intervention_interval in INTERVENTION_INTERVALS:
        for var in VARIANTS:
            grid = product(
                var["drop_strengths"],
                var["max_removal_fracs"],
                var["fixed_removal_fracs"],
                var["top_node_fracs"],
                var["drop_smoothing"],
                COMPLIANCES,
            )
            for ds, mrf, frem, tnf, dsmoothing, comp in grid:
                cnt += 1

                p = BASE_PARAMS.copy()
                p.update(
                    {
                        "intervention": True,
                        "removal_strategy": "partial",  # fixed-depth (future edges only)
                        "intervention_interval": intervention_interval,
                        "drop_strength": ds,
                        "drop_smoothing": dsmoothing,
                        "max_removal_fraction": mrf,
                        "fixed_removal_fraction": frem,
                        "top_node_frac": tnf,
                        "use_dynamic_threshold": var["use_dynamic_threshold"],
                        "fixed_threshold": var["fixed_threshold"],
                        "precomputed_max_risk": max_risk,
                        "compliance": comp,
                    }
                )

                suffix = make_suffix(
                    p,
                    ii=intervention_interval,
                    ds=ds,
                    mrf=mrf,
                    frem=frem,
                    tnf=tnf,
                )

                run_id = f"{var['label']}_{suffix}"
                params_fname = f"params_{run_id}.json"
                save_parameters(results_dir, params_fname, p)

                print(
                    f"\n[{cnt}/{inter_total} @ii={intervention_interval}] {var['label']} → "
                    f"λ={p['drop_strength']}, ϕ={p['fixed_removal_fraction']}, "
                    f"dsmoothing={p['drop_smoothing']} …"
                )

                results = run_sir_for_all_nodes(
                    temporal_network=temporal,
                    n_simulations=p["n_simulations"],
                    beta=p["beta"],
                    gamma=p["gamma"],
                    n_nodes=n_nodes,
                    scaling_exponent=p["scaling_exponent"],
                    intervention=True,
                    removal_strategy="partial",
                    threshold=p["threshold"],
                    window=p["window"],
                    intervention_interval=p["intervention_interval"],
                    n_timesteps=n_timesteps,
                    proportion_high_risk=p["proportion_high_risk"],
                    n_processes=p["n_processes"],
                    use_multiprocessing=p["use_multiprocessing"],
                    max_removal_fraction=p["max_removal_fraction"],
                    min_removal_fraction=p["min_removal_fraction"],
                    adjustment_step=p["adjustment_step"],
                    r0_change_window_size=p["r0_change_window_size"],
                    n_past_risk_values=p["n_past_risk_values"],
                    risk_model=p["risk_model"],
                    percentage_of_initial_infected=p["percentage_of_initial_infected"],
                    normalize_risk_score=p["normalize_risk_score"],
                    live_plot=p["live_plot"],
                    live_plot_callback=live_plot_callback,
                    precomputed_max_risk=p["precomputed_max_risk"],
                    rise_strength=p["rise_strength"],
                    drop_strength=p["drop_strength"],
                    accel_weight=p["accel_weight"],
                    pt_weight=p["pt_weight"],
                    rise_smoothing=p["rise_smoothing"],
                    drop_smoothing=p["drop_smoothing"],
                    use_dynamic_threshold=p["use_dynamic_threshold"],
                    fixed_threshold=p["fixed_threshold"],
                    fixed_removal_fraction=p["fixed_removal_fraction"],
                    top_node_frac=p["top_node_frac"],
                    is_baseline_run=p["is_baseline_run"],
                    compliance=p["compliance"],
                )

                zip_fname = f"results_{run_id}.zip"
                save_data(
                    os.path.join(results_dir, zip_fname),
                    {"partial": list(results.values())},
                    f"partial_{run_id}",
                )
                print(f"   ↳ saved → {zip_fname}")

                # θ = 0 baseline ("no-threshold" reference) with same λ, ϕ
                p_bl = p.copy()
                p_bl.update({"is_baseline_run": True})  # triggers θ = 0 in simulation

                suffix_bl = make_suffix(
                    p_bl,
                    ii=intervention_interval,
                    ds=ds,
                    mrf=mrf,
                    frem=frem,
                    tnf=tnf,
                )

                run_id_bl = f"{var['label']}_{suffix_bl}"
                save_parameters(results_dir, f"params_{run_id_bl}.json", p_bl)

                results_bl = run_sir_for_all_nodes(
                    temporal_network=temporal,
                    n_simulations=p_bl["n_simulations"],
                    beta=p_bl["beta"],
                    gamma=p_bl["gamma"],
                    n_nodes=n_nodes,
                    scaling_exponent=p_bl["scaling_exponent"],
                    intervention=True,
                    removal_strategy="partial",
                    threshold=p_bl["threshold"],
                    window=p_bl["window"],
                    intervention_interval=p_bl["intervention_interval"],
                    n_timesteps=n_timesteps,
                    proportion_high_risk=p_bl["proportion_high_risk"],
                    n_processes=p_bl["n_processes"],
                    use_multiprocessing=p_bl["use_multiprocessing"],
                    max_removal_fraction=p_bl["max_removal_fraction"],
                    min_removal_fraction=p_bl["min_removal_fraction"],
                    adjustment_step=p_bl["adjustment_step"],
                    r0_change_window_size=p_bl["r0_change_window_size"],
                    n_past_risk_values=p_bl["n_past_risk_values"],
                    risk_model=p_bl["risk_model"],
                    percentage_of_initial_infected=p_bl["percentage_of_initial_infected"],
                    normalize_risk_score=p_bl["normalize_risk_score"],
                    live_plot=p_bl["live_plot"],
                    live_plot_callback=live_plot_callback,
                    precomputed_max_risk=p_bl["precomputed_max_risk"],
                    rise_strength=p_bl["rise_strength"],
                    drop_strength=p_bl["drop_strength"],
                    accel_weight=p_bl["accel_weight"],
                    pt_weight=p_bl["pt_weight"],
                    rise_smoothing=p_bl["rise_smoothing"],
                    drop_smoothing=p_bl["drop_smoothing"],
                    use_dynamic_threshold=p_bl["use_dynamic_threshold"],
                    fixed_threshold=p_bl["fixed_threshold"],
                    fixed_removal_fraction=p_bl["fixed_removal_fraction"],
                    top_node_frac=p_bl["top_node_frac"],
                    is_baseline_run=p_bl["is_baseline_run"],
                    compliance=p_bl["compliance"],
                )

                zip_fname_bl = f"results_{run_id_bl}.zip"
                save_data(
                    os.path.join(results_dir, zip_fname_bl),
                    {"partial_thr0": list(results_bl.values())},
                    f"partial_thr0_{run_id_bl}",
                )
                print(f"   ↳ θ=0 baseline saved → {zip_fname_bl}")

    print("\nALL DONE.")


# ──────────────────────────────────────────────────────────────────────────────
# 5) Entry point ───────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROFILE = False
    if PROFILE:
        prof = cProfile.Profile()
        prof.enable()
        run_batch()
        prof.disable()
        pstats.Stats(prof).sort_stats("cumtime").print_stats(20)
    else:
        start = time.time()
        run_batch()
        dt = time.time() - start
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        print(f"\nTotal runtime: {int(h)}h {int(m)}m {s:.2f}s")
