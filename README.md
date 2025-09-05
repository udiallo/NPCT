# Network-based proactive contact tracing
*Network-based proactive contact tracing: A pre-emptive, degree-based alerting framework for privacy-preserving COVID-19 apps*  
Diaoulé Diallo, Tobias Hecking  
German Aerospace Center (DLR), Institute of Software Technology, Sankt Augustin, Germany

---

## Overview

This repository contains the code to reproduce the simulations and figures for our paper. We simulate SIR dynamics on **temporal contact networks** and apply a **dynamic-threshold intervention** with **fixed removal fractions**. 

---

## Repository layout

```
NPCT/
├─ data/                    # input datasets (ABM, DTU, Office are included here)
├─ notebooks_for_plots/     # notebooks to generate final paper figures
├─ paper_figures/           # exported figures for the paper
├─ plots/                   # live/static plots saved during runs
├─ data_utils.py
├─ graph_utils.py
├─ intervention.py
├─ main.py                  # entry point to run the simulations
├─ plotting_utils.py
├─ risk_computation.py
├─ simulation.py
└─ sir_functions.py
```

---

## Installation

Python ≥ 3.9 recommended.

Packages needed: numpy, pandas, matplotlib, networkx, tqdm

## Data

- **Included in `data/`:** ABM, DTU, and Office (workplace) networks used in the paper.
- **ABM30 (30-day extension):** download and place in `data/`  as micro_abm_contacts30.csv
  https://doi.org/10.5281/zenodo.15877149

**Original sources:**  
- Office (workplace): SocioPatterns — https://www.sociopatterns.org/datasets/office-proximity-network/  
- DTU (Copenhagen Networks Study, Bluetooth layer): Figshare — https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433  
- ABM: Zenodo — https://doi.org/10.5281/zenodo.15076221 (file: `TCN1000-medium.csv`)

---

## How to run simulations

1. **Select the network** in **`main.py`**:

   ```python
   BASE_PARAMS = {
       "network_name": "workplace",  # one of: "DTU" | "abm" | "abm30" | "workplace"
       ...
   }
   ```

2. (Optional) **Adjust the sweep settings** in `main.py`:

   ```python
   # Sensitivity λ (legacy name: drop_strength → “ds” in filenames/plots)
   DROP_STRENGTHS = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]

   # Fixed removal fractions ϕ
   FIXED_REMOVAL_FRACS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
   # For abm30, only the subset [0.10, 0.25, 0.50, 1.00] is needed
   ```

3. (Optional) **Multiprocessing**: can be disabled in `BASE_PARAMS`:

   ```python
   "use_multiprocessing": True,  # set to False to run single-process
   "n_processes": 30,            # adjust for your machine
   ```

4. **Run**:

   ```bash
   python main.py
   ```

Outputs (zipped results + parameter JSONs) will be written to the corresponding `results_<network>/` directory.

---

## Reproducing paper figures

- The notebooks in **`notebooks_for_plots/`** generate the figures used in the paper.  
- **Important:** run the simulations first (via `main.py`) so that the notebooks can load the produced results.

---

## Key modules

- **`simulation.py`** – runs SIR dynamics with optional interventions and returns a rich results dict per run.  
- **`intervention.py`** – dynamic-threshold logic, high-risk node selection, and fixed-depth (fractional) edge removals.  
- **`risk_computation.py`** – degree-based risk (paper default); NINL2/3/4 and ERM are available but not required.  
- **`sir_functions.py`** – SIR engine and network helpers (R₀ proxy, degree precomputation, edge lifespans).  
- **`data_utils.py`** – chunked saving/loading to `.zip` archives; JSON parameter I/O.  
- **`plotting_utils.py`** – static summary plotting and a multi-panel live plotting callback (headless safe via `Agg`).

---

---

## Notes

- The public results in the paper use **degree-based risk** with **dynamic threshold** + **fixed removal fraction**.
- To reduce runtime while testing, lower `n_simulations` in `BASE_PARAMS`.

