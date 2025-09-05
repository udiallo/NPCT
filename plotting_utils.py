# plotting_utils.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


def plot_results(results_dict):
    """Quick static plot: final number of infected across simulations."""
    final_infected = []
    for _, data in results_dict.items():
        final_states = data["history"][-1]  # last time step
        final_infected.append(int(np.sum(final_states == 1)))

    plt.figure(figsize=(8, 4))
    plt.title("Final number of infected across simulations")
    plt.plot(final_infected, "o-")
    plt.xlabel("Simulation ID")
    plt.ylabel("Num. Infected")
    plt.tight_layout()
    plt.show()


def live_plot_callback(
    data: dict,
    states: np.ndarray,
    removed_edges: set,
    adjacency_lists: list,
    current_step: int,
    intervention_step: int,
    risk_deques: dict,
    intervention: bool,
):
    """Live, multi-panel snapshot of simulation state (called periodically)."""
    n_intv = data["dynamic_thresholds"].shape[0] if "dynamic_thresholds" in data else 0
    intv_range = np.arange(n_intv)

    fig, axs = plt.subplots(4, 4, figsize=(22, 18))
    fig.suptitle(f"Live Plot at Intervention Step {current_step}", fontsize=18)

    # ── Row 1 ─────────────────────────────────────────────────────────────────

    # 1) SIR history
    if "history" in data:
        history = data["history"]  # (n_timesteps+1, n_nodes)
        time_steps = np.arange(history.shape[0])
        S_counts = np.sum(history == 0, axis=1)
        I_counts = np.sum(history == 1, axis=1)
        R_counts = np.sum(history == 2, axis=1)
        ax = axs[0, 0]
        ax.plot(time_steps, S_counts, label="S", color="blue")
        ax.plot(time_steps, I_counts, label="I", color="red")
        ax.plot(time_steps, R_counts, label="R", color="green")
        ax.set_title("SIR History")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Count")
        ax.legend()
    else:
        axs[0, 0].text(0.5, 0.5, "No history data", ha="center", va="center")

    # 2) Dynamic threshold (acceleration curve optional in data)
    if "dynamic_thresholds" in data and "acceleration" in data:
        ax = axs[0, 1]
        ax.plot(intv_range, data["dynamic_thresholds"], label="Dynamic Threshold", color="blue")
        ax.set_title("Dynamic Threshold")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("Value")
        ax.legend()
    else:
        axs[0, 1].text(0.5, 0.5, "No dynamic metrics", ha="center", va="center")

    # 3) R0 (raw & average)
    if "R0_values" in data and "R0_average_values" in data:
        ax = axs[0, 2]
        ax.plot(intv_range, data["R0_values"], label="Raw R₀", color="red")
        ax.plot(intv_range, data["R0_average_values"], label="Avg R₀", color="magenta", linestyle="--")
        ax.set_title("R₀ Dynamics")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("R₀")
        ax.legend()
    else:
        axs[0, 2].text(0.5, 0.5, "No R₀ data", ha="center", va="center")

    # 4) Edges removed per intervention
    if "edge_removal_history" in data:
        ax = axs[0, 3]
        ax.bar(intv_range, data["edge_removal_history"], color="gray")
        ax.set_title("Edge Removal History")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("Edges removed")
    else:
        axs[0, 3].text(0.5, 0.5, "No edge removal history", ha="center", va="center")

    # ── Row 2 ─────────────────────────────────────────────────────────────────

    # 5) Global infection potential P_t (with acceleration & product overlay)
    if "P_t" in data and "acceleration" in data:
        ax = axs[1, 0]
        ax.plot(intv_range, data["P_t"], marker="o", label="P_t", color="green")
        ax.plot(intv_range, data["acceleration"], label="Acceleration", color="orange", linestyle="--")
        combined = data["P_t"] * data["acceleration"]
        ax.plot(intv_range, combined, label="P_t × Acceleration", linestyle=":", color="purple")
        ax.set_title("Global Infection Potential & Acceleration")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("Value")
        ax.legend()
    else:
        axs[1, 0].text(0.5, 0.5, "No P_t/acceleration data", ha="center", va="center")

    # 6) Cumulative edge removal (% of original)
    if "cumulative_removed_edges" in data:
        ax = axs[1, 1]
        ax.plot(intv_range, data["cumulative_removed_edges"], marker="s", color="purple")
        ax.set_title("Cumulative Removed Edges (%)")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("% Removed")
    else:
        axs[1, 1].text(0.5, 0.5, "No cumulative removal data", ha="center", va="center")

    # 7) Count of high-risk nodes
    if "high_risk_node_counts" in data:
        ax = axs[1, 2]
        ax.plot(intv_range, data["high_risk_node_counts"], marker="^", color="brown")
        ax.set_title("High-Risk Node Counts")
        ax.set_xlabel("Intervention index")
        ax.set_ylabel("Count")
    else:
        axs[1, 2].text(0.5, 0.5, "No high-risk node counts", ha="center", va="center")

    # 8) Risk score histogram at current step
    if "risk_reduction_data" in data and "risk_scores" in data["risk_reduction_data"]:
        risk_scores = data["risk_reduction_data"]["risk_scores"]
        if risk_scores.shape[0] > intervention_step:
            current_risks = risk_scores[intervention_step, :]
            ax = axs[1, 3]
            ax.hist(current_risks, bins=20, edgecolor="black")
            mean_risk = float(current_risks.mean())
            ax.axvline(mean_risk, color="red", linestyle="--", label=f"Mean = {mean_risk:.2f}")
            ax.set_title("Risk Score Distribution")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Count")
            ax.legend()
        else:
            axs[1, 3].text(0.5, 0.5, "No risk data at current step", ha="center", va="center")
    else:
        axs[1, 3].text(0.5, 0.5, "No risk score data", ha="center", va="center")

    # ── Row 3 ─────────────────────────────────────────────────────────────────

    if "risk_reduction_data" in data and "risk_scores" in data["risk_reduction_data"]:
        risk_scores_arr = data["risk_reduction_data"]["risk_scores"]
        if risk_scores_arr.shape[0] >= n_intv and n_intv > 0:
            avg_risks = np.mean(risk_scores_arr, axis=1)

            final_scores = risk_scores_arr[-1]
            sorted_nodes = np.argsort(final_scores)
            top_20_idx = sorted_nodes[-20:]
            bottom_20_idx = sorted_nodes[:20]
            mid_start = max(0, len(sorted_nodes) // 2 - 10)
            middle_20_idx = sorted_nodes[mid_start : mid_start + 20]

            ax = axs[2, 0]
            ax.plot(intv_range, avg_risks, label="Avg Risk", color="blue")
            ax.set_title("Average Risk Score")
            ax.set_xlabel("Intervention index")
            ax.set_ylabel("Avg risk")
            ax.legend()

            ax = axs[2, 1]
            for i in top_20_idx:
                ax.plot(intv_range, risk_scores_arr[:, i], alpha=0.7)
            ax.set_title("Top 20 Risk Scores Over Time")
            ax.set_xlabel("Intervention index")
            ax.set_ylabel("Risk score")

            ax = axs[2, 2]
            for i in bottom_20_idx:
                ax.plot(intv_range, risk_scores_arr[:, i], alpha=0.7)
            ax.set_title("Bottom 20 Risk Scores Over Time")
            ax.set_xlabel("Intervention index")
            ax.set_ylabel("Risk score")

            ax = axs[2, 3]
            for i in middle_20_idx:
                ax.plot(intv_range, risk_scores_arr[:, i], alpha=0.7)
            ax.set_title("Middle 20 Risk Scores Over Time")
            ax.set_xlabel("Intervention index")
            ax.set_ylabel("Risk score")
        else:
            for j in range(4):
                axs[2, j].text(0.5, 0.5, "Not enough intervention steps", ha="center", va="center")
    else:
        for j in range(4):
            axs[2, j].text(0.5, 0.5, "No risk score data", ha="center", va="center")

    # ── Row 4 ─────────────────────────────────────────────────────────────────

    # 13) Edge counts over time
    if "edge_counts" in data:
        ax = axs[3, 0]
        edge_counts = data["edge_counts"]
        ax.plot(np.arange(len(edge_counts)), edge_counts, color="black")
        ax.set_title("Number of Edges Over Time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Edge count")
    else:
        axs[3, 0].text(0.5, 0.5, "No edge count data", ha="center", va="center")

    # 14) % edges remaining
    if "percentage_edges_per_snapshot_remaining" in data:
        ax = axs[3, 1]
        percentages = data["percentage_edges_per_snapshot_remaining"]
        ax.plot(np.arange(len(percentages)), percentages, color="teal")
        ax.set_title("Percentage of Edges Remaining")
        ax.set_xlabel("Time step")
        ax.set_ylabel("% remaining")
    else:
        axs[3, 1].text(0.5, 0.5, "No percentage data", ha="center", va="center")

    # 15) Degree time series (top/bottom 20)
    if adjacency_lists:
        n_nodes = len(states)
        n_snapshots = len(adjacency_lists)
        degrees = np.zeros((n_snapshots, n_nodes), dtype=int)

        for t, adj in enumerate(adjacency_lists):
            for node in range(n_nodes):
                deg = sum(
                    1
                    for neighbor in adj.get(node, [])
                    if (min(node, neighbor), max(node, neighbor), t) not in removed_edges
                )
                degrees[t, node] = deg

        final_degrees = degrees[-1]
        sorted_indices = np.argsort(final_degrees)
        top_20 = sorted_indices[-20:]
        bottom_20 = sorted_indices[:20]

        avg_deg = np.mean(degrees, axis=1)
        std_deg = np.std(degrees, axis=1)
        time_arr = np.arange(n_snapshots)

        ax = axs[3, 2]
        for i in top_20:
            ax.plot(time_arr, degrees[:, i], color="red", alpha=0.2)
        for i in bottom_20:
            ax.plot(time_arr, degrees[:, i], color="blue", alpha=0.2)
        ax.plot(time_arr, avg_deg, color="gray", linewidth=2, label="Avg degree")
        ax.fill_between(time_arr, avg_deg - std_deg, avg_deg + std_deg, color="gray", alpha=0.3, label="±1 std")
        ax.set_title("Degree Time Series (Top/Bottom 20)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Degree")
        ax.legend()
    else:
        axs[3, 2].text(0.5, 0.5, "No adjacency list", ha="center", va="center")

    # 16) Average degree: original vs. effective
    if adjacency_lists:
        n_nodes = len(states)
        n_snapshots = len(adjacency_lists)
        original_degrees = np.zeros(n_snapshots)
        effective_degrees = np.zeros(n_snapshots)

        for t, adj in enumerate(adjacency_lists):
            total_original = 0
            total_effective = 0
            for node in range(n_nodes):
                nbrs = adj.get(node, [])
                total_original += len(nbrs)
                total_effective += sum(
                    1
                    for neighbor in nbrs
                    if (min(node, neighbor), max(node, neighbor), t) not in removed_edges
                )
            original_degrees[t] = total_original / n_nodes
            effective_degrees[t] = total_effective / n_nodes

        time_arr = np.arange(n_snapshots)
        ax = axs[3, 3]
        ax.plot(time_arr, original_degrees, label="Original avg degree", color="black", linestyle="--")
        ax.plot(time_arr, effective_degrees, label="Effective avg degree", color="blue")
        ax.fill_between(
            time_arr,
            effective_degrees,
            original_degrees,
            where=original_degrees > effective_degrees,
            interpolate=True,
            color="red",
            alpha=0.2,
            label="Degree reduction",
        )
        ax.set_title("Avg Degree: Original vs. Effective")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Average degree")
        ax.legend()
    else:
        axs[3, 3].text(0.5, 0.5, "No adjacency list", ha="center", va="center")

    plt.tight_layout()

    out_path = (
        "./plots/latest_plot_intervention.png"
        if intervention
        else "./plots/latest_plot_no_intervention.png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
