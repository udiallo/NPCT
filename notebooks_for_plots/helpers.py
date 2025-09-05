import zipfile, pickle
import os
import re
from typing import Any, Dict, Tuple, List, Optional


import os, glob, time, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List, Any
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # for the sigmoid fit
from matplotlib.lines import Line2D
import pandas as pd
import re
from matplotlib import cm


def _to_frac(raw: str) -> Optional[float]:
    """
    Convert a percentage‐style token into a float:
      - "NA"    → None
      - digits  → int(raw)/100.0   (e.g. "075" → 0.75)
    """
    if raw == "NA":
        return None
    return int(raw) / 100.0

def _to_number(raw: str) -> float:
    """
    Convert a 'p'‐delimited token into a float, or plain digits:
      - "1p0"   → 1.0
      - "12"    → 12.0
    """
    return float(raw.replace("p", "."))

def parse_fn(filename: str) -> Optional[Tuple[Any, ...]]:
    """
    Parse filenames of the form

        results_<variant>_iiXX_dsYY_mrfZZ_fremWW_tnfVV_ftTrue_...

    NEW: the `_tnfVV` token (top-node-fraction).  See doc-string for return-order.
    """

    stem = os.path.splitext(os.path.basename(filename))[0]
    if not stem.startswith("results_"):
        return None
    stem = stem[len("results_"):]

    is_baseline = stem.endswith("_BASELINE")
    if is_baseline:
        stem = stem[:-len("_BASELINE")]

    pattern = (
        r'^(?P<variant>.+?)'
        r'_ii(?P<ii>\d+)'
        r'_ds(?P<ds>[0-9NA]+)'
        r'_mrf(?P<mrf>[0-9NA]+)'
        r'_frem(?P<frem>[0-9NA]+)'
        r'_tnf(?P<tnf>[0-9NA]+)'        
        r'(?:_c(?P<c_pre>[0-9]{2,3}))?'
        r'(?:_ibl(?P<ibl>[01]))?'  
        r'(?:_c(?P<c_post>[0-9]{2,3}))?'
        r'_ft(?P<ft>True|False)'
        r'_nprv(?P<nprv>\d+)'
        r'_rm(?P<rm>[^_]+)'
        r'_w(?P<w>[0-9p]+)'
        r'_aw(?P<aw>[0-9p]+)'
        r'_pt(?P<pt>[0-9p]+)'
        r'_rs(?P<rs>[0-9p]+)'
        r'_dsp(?P<dsp>[0-9p]+)'
        r'_b(?P<b>[0-9p]+)'
        r'_g(?P<g>[0-9p]+)'
        r'$'
    )
    m = re.match(pattern, stem)
    if not m:
        return None

    raw_c = m.group("c_pre") or m.group("c_post") or "100"
    comp  = _to_frac(raw_c)

    variant      = "without" if is_baseline else m.group("variant")
    ii           = int(m.group("ii"))
    ds           = _to_frac(m.group("ds"))
    mrf          = _to_frac(m.group("mrf"))
    frem         = _to_frac(m.group("frem"))
    tnf          = _to_frac(m.group("tnf"))        # ← NEW
    ibl_flag     = bool(int(m.group("ibl") or "0"))
    # comp         = _to_frac(m.group("c") or "100")
    ft_flag      = (m.group("ft") == "True")
    nprv         = int(m.group("nprv"))
    risk_model   = m.group("rm")

    window       = _to_number(m.group("w"))
    accel_weight = _to_number(m.group("aw"))
    pt_weight    = _to_number(m.group("pt"))
    rise_smooth  = _to_number(m.group("rs"))
    drop_smooth  = _to_number(m.group("dsp"))
    beta         = _to_number(m.group("b"))
    gamma        = _to_number(m.group("g"))

    # return-order grew by one (tnf)
    return (
        variant, ii, ds, mrf, frem, tnf, comp, ibl_flag,
        ft_flag, nprv, risk_model,
        window, accel_weight, pt_weight,
        rise_smooth, drop_smooth,
        beta, gamma,
        is_baseline
    )



def agg_basic(runs):
    """Compute population, mean final infected, mean edges-removed (%)."""
    pop = runs[0]["history"].shape[1]

    finf, erem_pct = [], []
    for rd in runs:
        I = np.count_nonzero(rd["history"][-1] == 1)
        R = np.count_nonzero(rd["history"][-1] == 2)
        finf.append(I + R)

        raw = rd['cumulative_removed_edges'][-1] 
        raw_pct = float(raw)
        erem_pct.append(raw_pct)

    return {
        "pop": pop,
        "finf": np.mean(finf),
        "erem_pct": np.mean(erem_pct),     # always in %
    }


class FixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # remap any numpy._core or numpy._core.<sub> → numpy.core or numpy.core.<sub>
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_runs(zp_path):
    """Load the runs dict from a .zip via FixUnpickler."""
    with zipfile.ZipFile(zp_path, 'r') as zf:
        pkl = next(n for n in zf.namelist() if n.endswith('.pkl'))
        with zf.open(pkl) as fh:
            data = FixUnpickler(fh).load()
    return list(data.values())[0]


def matches_filters(row, filters):
    if filters.get("variants") is not None and row["variant"] not in filters["variants"]:
        return False
    if filters.get("ds") is not None and row["drop_strength"] not in filters["ds"]:
        return False
    if filters.get("fixed_frac") is not None and row["fixed_frac"] not in filters["fixed_frac"]:
        return False
    if filters.get("max_rf") is not None and row["max_RF"] not in filters["max_rf"]:
        return False
    if filters.get("top_node_frac") is not None and row["top_node_frac"] not in filters["top_node_frac"]:
        return False                                # ← NEW
    if filters.get("ii") is not None and row["ii"] not in filters["ii"]:
        return False
    if row["variant"] != "without":
        if filters.get("compliance") is not None and row["compliance"] not in filters["compliance"]:
            return False
    return True



def collect_summaries(df, cache):
    """
    Build dict label→summary for plotting, now including top_node_frac in the key
    so that each (variant, cap, tnf, ds) becomes its own entry.
    """
    summ = {}

    for _, row in df.iterrows():
        variant = row["variant"]
        ds      = float(row["drop_strength"])
        mrf     = None if pd.isna(row.get("max_RF", None))    else float(row["max_RF"])
        cap     = None if pd.isna(row.get("fixed_frac", None)) else float(row["fixed_frac"])
        tnf     = None if pd.isna(row.get("top_node_frac", None)) else float(row["top_node_frac"])
        ii      = None if pd.isna(row.get("ii", None))         else int(row["ii"])
        ft_flag = bool(row.get("fixed_threshold", False))
        ibl_flag = bool(row.get("intervention_baseline", False))
        comp = None if pd.isna(row.get("compliance")) else float(row["compliance"])

        # --- find the *exact* cache key matching all six dimensions ----
        full_key = None
        for k in cache:
            # k is a tuple like
            # (variant, ii, ds, mrf, cap, tnf, ft_flag, nprv, rm, w, aw, pt, rs, dsp, b, g)
            if (k[0] == variant and
                k[1] == ii      and
                k[2] == ds      and
                k[3] == mrf     and
                k[4] == cap     and
                k[5] == tnf     and
                k[6] == comp    and
                k[7] == ibl_flag     and
                k[8] == ft_flag):
                full_key = k
                break

        if full_key is None:
            print(f"⚠️  no summary found for {variant}, ds={ds}, cap={cap}, tnf={tnf}")
            continue

        # --- build a unique label including cap, tnf, ds  --------------------
        parts = [variant]
        if cap is not None:
            parts.append(f"cap{cap:.2f}")
        if tnf is not None:
            parts.append(f"tnf{tnf:.2f}")
        parts.append(f"ds{ds:.2f}")
        lbl = "_".join(parts)

        summ[lbl] = cache[full_key]

    return summ


def summarise_fixeddepth(runs, n_bins: int = 20):
    """
    Aggregate simulation runs for one (ds, cap) variant.

    Returns a dict with:
      • edges_remaining_mean / min / max      (len = n_timesteps)
      • avg/min/max_high_risk_node_counts     (len = n_interventions)
      • new_high_risk_mean / min / max        (len = n_interventions)
      • people_targeted_pct                   (scalar)
      • threshold_mean / threshold_std        (len = n_interventions)
      • risk_mean / risk_std / risk_med       (len = n_interventions)
      • risk_bins                             (len = n_bins+1)
      • risk_hist_mean / risk_hist_std / risk_hist_med
                                              (shape (n_interventions, n_bins))
      • acceleration_mean / acceleration_std / acceleration_med
      • P_t_mean / P_t_std / P_t_med
      • … plus all your existing fields …
    """
    import numpy as np

    # 1) Edges remaining over time
    edges = np.stack([r["percentage_edges_per_snapshot_remaining"] for r in runs])
    edges_mean = edges.mean(axis=0)
    edges_min  = edges.min(axis=0)
    edges_max  = edges.max(axis=0)
    edges_std  = edges.std(axis=0)   # ← new

    # 2) High-risk node counts
    hi = np.stack([r.get("high_risk_node_counts", np.zeros_like(edges[0])) for r in runs])

    # 3) New high-risk counts per intervention
    def new_highrisk_counts(run):
        prev = set(); out = []
        for s in run["high_risk_nodes"]:
            cur = set(s)
            out.append(len(cur - prev))
            prev = cur
        return np.array(out, dtype=int)
    delta_hi = np.stack([new_highrisk_counts(r) for r in runs])

    # 4) % people ever targeted
    pop = runs[0]["history"].shape[1]
    union = set()
    for r in runs:
        union |= set().union(*r["high_risk_nodes"])
    people_targeted_pct = 100 * len(union) / pop

    # 5) Dynamic-threshold stats
    th = np.stack([r["dynamic_thresholds"] for r in runs])  # (n_runs, n_int)
    threshold_mean = th.mean(axis=0)
    threshold_std  = th.std(axis=0)

    # 6) Risk-score evolution stats
    run_means = np.stack([r["risk_scores_over_time"].mean(axis=1) for r in runs])
    risk_mean = run_means.mean(axis=0)
    risk_std  = run_means.std(axis=0)
    risk_med  = np.median(run_means, axis=0)

    # 7) Risk-score distributions (histograms)
    n_int = run_means.shape[1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    hist = np.zeros((len(runs), n_int, n_bins), dtype=float)
    for i, r in enumerate(runs):
        scores = r["risk_scores_over_time"]
        for t in range(n_int):
            h, _ = np.histogram(scores[t, :], bins=bins)
            hist[i, t, :] = h
    risk_hist_mean = hist.mean(axis=0)
    risk_hist_std  = hist.std(axis=0)
    risk_hist_med  = np.median(hist, axis=0)

    # 8) Final per-node burden & risk
    edges_removed_runs   = [r["risk_reduction_data"]["edges_reduced"].sum(axis=0) for r in runs]
    edges_removed_per_node = np.mean(edges_removed_runs, axis=0)
    risk_final_runs      = [r["risk_scores_over_time"][-1] for r in runs]
    risk_final           = np.mean(risk_final_runs, axis=0)

    timeavg_per_run = [r["risk_scores_over_time"].mean(axis=0) for r in runs]
    risk_timeavg = np.mean(timeavg_per_run, axis=0)

    ever_high_prob = np.mean([
        np.isin(np.arange(risk_final.size),
                list(set().union(*r["high_risk_nodes"]))).astype(float)
        for r in runs
    ], axis=0)
    ever_high_mask = ever_high_prob > 0.5

    # 9) Per-step incremental burden & per-step risk averages
    edges_incr = np.stack([r["risk_reduction_data"]["edges_reduced"] for r in runs], axis=0)
    edges_incr_mean = edges_incr.mean(axis=0)

    risk_scores_over_time_mean = np.stack(
        [r["risk_scores_over_time"] for r in runs], axis=0
    ).mean(axis=0)

    removal_frac_mean = np.stack(
        [r["risk_reduction_data"]["removal_fractions"] for r in runs],
        axis=0
    ).mean(axis=0)

     # 10) Mean infections over time
    infected_ts    = np.stack([(run["history"] == 1).sum(axis=1) for run in runs])  # shape (n_runs, T)
    infected_mean  = infected_ts.mean(axis=0)
    infected_std   = infected_ts.std(axis=0)          # ← new
    infected_med   = np.median(infected_ts, axis=0)   # ← optional

    # Per-run final infected counts (I+R at last time step)
    final_infected_abs_runs = np.array([
        (run["history"][-1] == 1).sum() + (run["history"][-1] == 2).sum()
        for run in runs
    ], dtype=int)



    # 11) Acceleration & potential stats
    #    Assumes r["acceleration"] and r["P_t"] are arrays of length n_int
    acc = np.stack([r.get("acceleration", np.zeros(n_int)) for r in runs])
    acceleration_mean = acc.mean(axis=0)
    acceleration_std  = acc.std(axis=0)
    acceleration_med  = np.median(acc, axis=0)

    Pt = np.stack([r.get("P_t", np.zeros(n_int)) for r in runs])
    P_t_mean = Pt.mean(axis=0)
    P_t_std  = Pt.std(axis=0)
    P_t_med  = np.median(Pt, axis=0)

    # ———————————————————————————————————————————————————————————
    # Compute per-node high-risk selection counts
    # ———————————————————————————————————————————————————————————
    # runs[i]["high_risk_nodes"] is a list of length n_interventions,
    # each entry a list (or set) of node indices flagged at that step.
    n_runs = len(runs)
    n_nodes = runs[0]["history"].shape[1]
    # accumulate a (n_runs × n_nodes) matrix of counts
    sel_counts = np.zeros((n_runs, n_nodes), dtype=int)
    for i, r in enumerate(runs):
        for step_nodes in r["high_risk_nodes"]:
            # turn the iterable of node‐IDs into an integer array
            idx = np.asarray(list(step_nodes), dtype=int)
            if idx.size:
                sel_counts[i, idx] += 1
    # average across runs
    sel_count_per_node = sel_counts.mean(axis=0)

    edges_removed_pct_runs = np.array([
        float(r['cumulative_removed_edges'][-1]) for r in runs
    ])
    
    # 12) Package summary
    return {

        "sel_count_per_node": sel_count_per_node,

        # edges remaining
        "edges_remaining_mean": edges_mean,
        "edges_remaining_std":  edges_std,   # ← new
        "edges_remaining_min":  edges_min,
        "edges_remaining_max":  edges_max,

        # high-risk counts
        "avg_high_risk_node_counts": hi.mean(axis=0),
        "min_high_risk_node_counts": hi.min(axis=0),
        "max_high_risk_node_counts": hi.max(axis=0),

        # new high-risk
        "new_high_risk_mean": delta_hi.mean(axis=0),
        "new_high_risk_min":  delta_hi.min(axis=0),
        "new_high_risk_max":  delta_hi.max(axis=0),

        # population targeted
        "people_targeted_pct": people_targeted_pct,

        # dynamic threshold
        "threshold_mean": threshold_mean,
        "threshold_std":  threshold_std,

        # risk evolution
        "risk_mean": risk_mean,
        "risk_std":  risk_std,
        "risk_med":  risk_med,

        # risk distributions
        "risk_bins":      bins,
        "risk_hist_mean": risk_hist_mean,
        "risk_hist_std":  risk_hist_std,
        "risk_hist_med":  risk_hist_med,

        # backward compatibility
        "risk_count_mean": risk_hist_mean,
        "risk_count_std":  risk_hist_std,

        # final per-node
        "edges_removed_per_node": edges_removed_per_node,
        "risk_final":             risk_final,
        "risk_timeavg":           risk_timeavg,

        # backward compatibility
        "ever_high_mask": ever_high_mask,

        # step-wise arrays
        "edges_incr_mean":            edges_incr_mean,
        "removal_frac_mean":          removal_frac_mean,
        "risk_scores_over_time_mean": risk_scores_over_time_mean,
        "infected_mean": infected_mean,
        "infected_std":  infected_std,     # ← new
        "infected_med":  infected_med,     # ← optional
        "infected_ts": infected_ts, 

        # acceleration & potential
        "acceleration_mean": acceleration_mean,
        "acceleration_std":  acceleration_std,
        "acceleration_med":  acceleration_med,
        "P_t_mean":          P_t_mean,
        "P_t_std":           P_t_std,
        "P_t_med":           P_t_med,

        "edges_removed_pct_runs": edges_removed_pct_runs,
        "final_infected_abs_runs": final_infected_abs_runs,
        "final_infected_pct_runs": 100.0 * final_infected_abs_runs / pop,
    }

EXTRA = {
    "peak_infected_pct": lambda runs: 100 * np.mean([
        np.max(np.sum(rd['history'] == 1, axis=1)) / rd['history'].shape[1]
        for rd in runs
    ]),
    "time_to_peak": lambda runs: np.mean([
        np.argmax(np.sum(rd['history'] == 1, axis=1))
        for rd in runs
    ]),
    "auc_I_pct": lambda runs: 100 * np.mean([
        np.trapz(np.sum(rd['history'] == 1, axis=1)) /
        (rd['history'].shape[1] * len(rd['history']))
        for rd in runs
    ]),
    "mean_threshold": lambda runs: np.mean([
        np.mean(rd['dynamic_thresholds']) for rd in runs
    ]),
    "wall_time_s": lambda runs: np.mean([
        rd['timing_info']['total_time'] for rd in runs
    ]),
}

def _build_or_load(CACHE_DIR: str, zp_path: str):
    """
    Load one ZIP (or its pickle cache) and return
      msg: str,
      tup: (key, rec),
      summary: Dict with all your pre-aggregated data.

    Keys are always the 15-tuple from make_key():
      (variant, ii, ds, mrf, frem,
       ft_flag, nprv, rm, w, aw,
       pt, rs, dsp, b, g)
    """
    name = os.path.basename(zp_path)
    base = os.path.splitext(name)[0]
    pkl  = os.path.join(CACHE_DIR, base + ".pkl")

    # ── Fast path: try loading cached pickle ────────────────────────────────
    if os.path.exists(pkl):
        try:
            key_loaded, rec, summary = pickle.load(open(pkl, "rb"))

            # ── Backwards‐compat: old pickles used a 15‐tuple (no tnf) ──────────
            flen = len(key_loaded)
            if flen < 18:
                kl = list(key_loaded)

                if flen == 15:            # no tnf, no comp, no ibl
                    kl.insert(5, None)    # tnf
                    kl.insert(6, 1.0)     # comp
                    kl.insert(7, False)   # ibl
                elif flen == 16:          # has tnf, lacks comp & ibl
                    kl.insert(6, 1.0)
                    kl.insert(7, False)
                elif flen == 17:          # has tnf & ibl, lacks comp
                    kl.insert(6, 1.0)

                key_loaded = tuple(kl)

            # this is now our canonical 16‐tuple key
            key = key_loaded
            migrated = False

            # ── Ensure rec has the new field ───────────────────────────────────
            if "top_node_frac" not in rec:
                rec["top_node_frac"] = key[5]
                migrated = True

            # ── Your existing migration steps ─────────────────────────────────
            if "people_targeted_pct" not in rec:
                rec["people_targeted_pct"] = summary["people_targeted_pct"]
                migrated = True
            
            if "intervention_baseline" not in rec:
                rec["intervention_baseline"] = bool(key_loaded[7])
                migrated = True
            
            if "compliance" not in rec:        
                rec["compliance"] = key_loaded[6]

            if any(k not in summary for k in (
            "new_high_risk_mean", "new_high_risk_min", "new_high_risk_max"
            )):
                runs    = load_runs(zp_path)
                summary = summarise_fixeddepth(runs)
                migrated = True
                rec["people_targeted_pct"] = summary["people_targeted_pct"]

            # ── Rewrite cache if we added anything ─────────────────────────────
            if migrated:
                with open(pkl, "wb") as f:
                    pickle.dump((key, rec, summary),
                                f, protocol=pickle.HIGHEST_PROTOCOL)

            return f"✓ {name} (from cache)", (key, rec), summary

        except Exception:
            # fall back to slow path on any error
            pass


    # ── Slow path: parse filename and rebuild from runs ────────────────────
    meta = parse_fn(name)
    if meta is None:
        return f"⚠︎ skipping unparseable {name}", (), None

    # meta is 16 values; unpack and drop the last (is_baseline) into _
    (variant, ii, ds, mrf, frem, tnf, comp, ibl_flag, ft_flag,
    nprv, rm, w, aw, pt, rs, dsp, b, g, is_baseline) = meta

    # Load raw simulation runs
    runs  = load_runs(zp_path)
    basic = agg_basic(runs)

    # compute % people ever targeted
    pop = basic["pop"]
    nodes_union = set()
    for run in runs:
        nodes_union |= set.union(*run["high_risk_nodes"])
    people_targeted_pct = 100 * len(nodes_union) / pop

    # Build rec dict
    rec = {
        "variant":             variant,
        "ii":                  ii,
        "drop_strength":       ds,
        "max_RF":              mrf,
        "fixed_frac":          frem,
        "top_node_frac":       tnf,
        "fixed_threshold":     ft_flag,
        "n_past_risk_values":  nprv,
        "risk_model":          rm,
        "window":              w,
        "accel_weight":        aw,
        "pt_weight":           pt,
        "rise_smoothing":      rs,
        "drop_smoothing":      dsp,
        "beta":                b,
        "gamma":               g,
        "finf_abs":            basic["finf"],
        "pop":                 pop,
        "edges_removed_pct":   basic["erem_pct"],
        "final_infected_pct":  100 * basic["finf"] / pop,
        "people_targeted_pct": people_targeted_pct,
        "intervention_baseline": bool(ibl_flag),
        "compliance": comp,
    }
    # Add any EXTRA fields
    for k, fn in EXTRA.items():
        rec[k] = fn(runs)

    # Build summary
    summary = summarise_fixeddepth(runs)

    # Build the 15-tuple key and cache it
    key = make_key(
        variant, ii, ds, mrf, frem, tnf, comp, ibl_flag,
        ft_flag, nprv, rm, w, aw, pt, rs, dsp, b, g
    )
    with open(pkl, "wb") as f:
        pickle.dump((key, rec, summary),
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    return f"✓ {name}", (key, rec), summary


def load_fixeddepth_summary(RESULTS_DIR: str,
                            CACHE_DIR: str,
                            filters: Dict[str, Any],
                            limit_fixed: int = None):
    """
    Return:
      df: DataFrame with all recs (baselines + fixed-removal + risk-depth)
      cache: dict key→summary

    We now apply the same filtering to baseline files as to any other run,
    so that only the baseline with matching (beta, gamma, etc.) gets picked.
    """

    all_zips = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.zip")))

    # First, filter EVERY zip (including baselines) by exactly the same set of criteria in `filters`.
    filtered_zips = []
    for zp in all_zips:
        stem = os.path.basename(zp)
        meta = parse_fn(stem)
        if meta is None:
            continue

        (variant, ii, ds, mrf, frem, tnf, comp, ibl_flag, ft_flag,
            nprv, rm, w, aw, pt, rs, dsp, b, g, is_baseline) = meta

        # Apply every filter in `filters` (including beta/gamma) to this run.
        if filters.get("variants") is not None and variant not in filters["variants"]:
            continue
        if filters.get("ds") is not None and ds not in filters["ds"]:
            continue
        if filters.get("max_rf") is not None and mrf not in filters["max_rf"]:
            continue
        if filters.get("fixed_frac") is not None and frem not in filters["fixed_frac"]:
            continue
        if filters.get("ii") is not None and ii not in filters["ii"]:
            continue
        if filters.get("fixed_threshold") is not None and ft_flag not in filters["fixed_threshold"]:
            continue
        if filters.get("nprv") is not None and nprv not in filters["nprv"]:
            continue
        if filters.get("risk_model") is not None and rm not in filters["risk_model"]:
            continue
        if filters.get("window") is not None and w not in filters["window"]:
            continue
        if filters.get("accel_weight") is not None and aw not in filters["accel_weight"]:
            continue
        if filters.get("pt_weight") is not None and pt not in filters["pt_weight"]:
            continue
        if filters.get("rise_smoothing") is not None and rs not in filters["rise_smoothing"]:
            continue
        if filters.get("drop_smoothing") is not None and dsp not in filters["drop_smoothing"]:
            continue
        if filters.get("beta") is not None and b not in filters["beta"]:
            continue
        if filters.get("gamma") is not None and g not in filters["gamma"]:
            continue
        if filters.get("top_node_frac") is not None and tnf not in filters["top_node_frac"]:
            continue
        if filters.get("ibl") is not None and ibl_flag not in filters["ibl"]:
            continue
        if filters.get("compliance") is not None and comp not in filters["compliance"]:
            continue

        # If we get here, this (possibly “without”) run passes all filters.
        filtered_zips.append(zp)

    if not filtered_zips:
        raise RuntimeError("No files remain after applying your filters!")

    # 2) split into baselines / risk_depth / fixed_depth (unchanged) …
    baselines, risk_depth, fixed_depth = [], [], []
    for zp in filtered_zips:
        (variant, ii, ds, mrf, frem, tnf, comp, ibl_flag, ft_flag,
         nprv, rm, w, aw, pt, rs, dsp, b, g,
         is_baseline) = parse_fn(os.path.basename(zp))
        if is_baseline or variant == "without":
            baselines.append(zp)
        elif frem is None:
            risk_depth.append(zp)
        else:
            fixed_depth.append(zp)

    risk_depth  = risk_depth[:limit_fixed]
    fixed_depth = fixed_depth[:limit_fixed]
    targets = baselines + risk_depth + fixed_depth

    print(f"Loading {len(baselines)} baseline + "
          f"{len(risk_depth)} risk-depth + {len(fixed_depth)} fixed-depth files")

    # 3) load each ZIP (or cache) in parallel, record both rec and summary
    rows, cache = [], {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 50)) as pool:
        futs = {pool.submit(_build_or_load, CACHE_DIR, zp): zp for zp in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            zp = futs[fut]
            msg, tup, summary = fut.result()
            print(f"{msg}   ({i}/{len(targets)})")
            if not tup:
                continue
            key, rec = tup
            (variant, ii, ds, mrf, frem, tnf, comp, ibl_flag, ft_flag,
             nprv, rm, w, aw, pt, rs, dsp, b, g) = key

            rec["fixed_threshold"] = ft_flag
            rec.pop("mode", None)
            rec.update({
                "n_past_risk_values": nprv,
                "risk_model":         rm,
                "window":             w,
                "accel_weight":       aw,
                "pt_weight":          pt,
                "rise_smoothing":     rs,
                "drop_smoothing":     dsp,
                "beta":               b,
                "gamma":              g,
            })
            rec["method"] = "risk-depth" if frem is None else "fixed-removal"

            # **store the original key** so we can index cache later
            rec["_key"] = key

            rows.append(rec)
            cache[key] = summary

    # 4) build DataFrame
    df = pd.DataFrame(rows)

    # 5) compute baseline-peak time & value per (ds, tnf)
    pop = df.loc[df.variant == "without", "pop"].iat[0]
    base_time_idx, base_time_val = {}, {}
    for key, summ in cache.items():
        if key[0] != "without":
            continue
        ds_, tnf_ = key[2], key[5]
        inf_series = np.asarray(summ["infected_mean"]) / pop * 100
        idx = int(np.argmax(inf_series))
        base_time_idx[(ds_, tnf_)] = idx
        base_time_val[(ds_, tnf_)] = inf_series[idx]

    
    # fallback maps ignoring tnf
    base_time_idx_ds = {ds_: idx for (ds_, _), idx in base_time_idx.items()}
    base_time_val_ds = {ds_: val for (ds_, _), val in base_time_val.items()}

    # 6) compute reduction at baseline peak time
    def _compute_baseline_reduction(row):
        ds_  = row["drop_strength"]
        tnf_ = row["top_node_frac"]
        idx  = base_time_idx.get((ds_, tnf_),
               base_time_idx_ds.get(ds_, None))
        if idx is None:
            return np.nan
        infected = np.asarray(cache[row["_key"]]["infected_mean"]) / row["pop"] * 100
        base_val = base_time_val.get((ds_, tnf_),
                   base_time_val_ds.get(ds_, np.nan))
        return 100.0 * (base_val - infected[idx]) / base_val

    df["peak_reduced_pct"] = df.apply(_compute_baseline_reduction, axis=1)

    # def _compute_peak_std(row):
    #     ds, tnf = row["drop_strength"], row["top_node_frac"]
    #     idx = base_time_idx.get((ds, tnf), base_time_idx_ds[ds])
    
    #     ts   = cache[row["_key"]]["infected_ts"]   # (n_runs, T)
    #     pop  = row["pop"]
    #     base = base_time_val.get((ds, tnf), base_time_val_ds[ds])
    
    #     pct  = ts[:, idx] / pop * 100              # % infected per run
    #     red  = 100 * (base - pct) / base           # reduction per run
    #     return float(np.std(red, ddof=1))          # unbiased σ
    
    # df["peak_reduced_std"] = df.apply(_compute_peak_std, axis=1)

    # 7) compute old peak_reduced_pct (overall) exactly as before
    base_peaks_exact = (
        df[df.variant == "without"]
          .groupby(["drop_strength","top_node_frac"])["peak_infected_pct"]
          .mean()
          .to_dict()
    )
    base_peaks_ds = (
        df[df.variant == "without"]
          .groupby("drop_strength")["peak_infected_pct"]
          .mean()
          .to_dict()
    )

    df["overall_peak_reduced_pct"] = df.apply(
        lambda r: 100 * (
            base_peaks_exact.get((r.drop_strength, r.top_node_frac),
                                base_peaks_ds.get(r.drop_strength, np.nan))
            - r.peak_infected_pct
        ) / base_peaks_ds.get(r.drop_strength, np.nan),
        axis=1
    )

    # 8) infection-reduction % and basic efficiencies (unchanged)
    mask = df.variant == "without"
    base_inf = df.loc[mask, "finf_abs"].mean()
    df["infection_reduced_pct"] = 100 * (base_inf - df["finf_abs"]) / base_inf

    df["efficiency"]        = np.where(df["edges_removed_pct"] > 0.05,
                                       df["infection_reduced_pct"] / df["edges_removed_pct"], 0.0)
    df["people_efficiency"] = np.where(df["people_targeted_pct"] > 0.05,
                                       df["infection_reduced_pct"] / df["people_targeted_pct"], 0.0)

    # 9) peak efficiencies: old vs. new
    df["peak_efficiency"] = np.where(
        df["edges_removed_pct"] > 0.05,
        df["peak_reduced_pct"] / df["edges_removed_pct"],
        0.0
    )

    df["overall_peak_efficiency"] = np.where(
        df["edges_removed_pct"] > 0.05,
        df["overall_peak_reduced_pct"] / df["edges_removed_pct"],
        0.0
    )

    df["people_peak_efficiency"] = np.where(
        df["people_targeted_pct"] > 0.05,
        df["peak_reduced_pct"] / df["people_targeted_pct"],
        0.0
    )

    df["people_overall_peak_efficiency"] = np.where(
        df["people_targeted_pct"] > 0.05,
        df["overall_peak_reduced_pct"] / df["people_targeted_pct"],
        0.0
    )

    return df, cache



def marker_map(drop_strengths):
    seq = ["o","s","^","D","P","X","*","h"]
    real_ds = [d for d in drop_strengths if not pd.isna(d)]
    base = {d: seq[i % len(seq)] for i, d in enumerate(sorted(real_ds))}
    base[np.nan] = "o"
    return base

def make_color_map(values, cmap_name="viridis"):
    # ignore NaNs when normalising
    clean = [v for v in values if not pd.isna(v)]
    if not clean:                            # all NaN  → dummy range
        clean = [0, 1]
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(min(clean), max(clean))

    def _col(v):
        return "white" if pd.isna(v) else cmap(norm(v))

    return _col, cm.ScalarMappable(norm=norm, cmap=cmap)


def make_key(variant, ii, ds, mrf, frem, tnf, comp, ibl_flag,
             ft_flag, nprv, rm, w, aw, pt, rs, dsp, b, g):
    return (
        variant, ii, ds, mrf, frem, tnf, comp, ibl_flag,
        ft_flag, nprv, rm, w, aw, pt, rs, dsp, b, g
    )