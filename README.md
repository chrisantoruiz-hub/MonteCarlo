# Monte Carlo Beamline Simulator v2

Browser-based Monte Carlo simulator for particle trajectories through a two-MCP beamline with grids, tilted wire planes, and an ionisation chamber. FastAPI backend + vanilla JS + Plotly.js frontend.

---

## Quick start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Open in browser:
open ../frontend/index.html
```

---

## Geometry

All z positions are relative to grid1. Aperture is **5 × 5 cm** (±25 mm) at every element.

### MCP1 (z = 0 – 0.037 m)

| Element | z (m)  | Description |
|---------|--------|-------------|
| grid1   | 0.000  | MN4 square mesh: pitch 1238 µm, t = 32 µm (x- and y-wires) |
| WP1     | 0.010  | Plane **⊥ to beam**; vertical wires (blocking coord u = x_local) |
| WP2     | 0.035  | Plane **tilted +45° to z-axis** (in x-z plane); vertical wires |
| WP3     | 0.037  | Plane **tilted +45° to z-axis**; vertical wires; parallel to WP2, 2 mm downstream |

### MCP2 (z = 0.481 – 0.520 m)

| Element | z (m)  | Description |
|---------|--------|-------------|
| WP4     | 0.481  | Plane **tilted −45° to z-axis** (in x-z plane); vertical wires; first in beam order |
| WP5     | 0.483  | Plane **tilted −45° to z-axis**; vertical wires; parallel to WP4, 2 mm downstream |
| WP6     | 0.508  | Plane **⊥ to beam**; vertical wires (blocking coord u = x_local) |
| grid2   | 0.520  | MN8 square mesh: pitch 803 µm, t = 43 µm (x- and y-wires) |

### Downstream

| Element | z (m)  | Description |
|---------|--------|-------------|
| IC      | 0.620  | Ionisation chamber, efficiency η_IC, same ±2.5 cm aperture |

### Wire plane common parameters

- Pitch = 1000 µm, wire diameter = 20 µm (cylindrical) → T = 98 % per plane (1-D analytic)
- All wires are **horizontal** (run along x; block in y, u = y_local). The ±45° refers to the tilt of the **physical plane** relative to the beam axis, not the wire orientation.
- For the wire-hit check, the nominal-z approximation is used: the x-dependent shift in crossing-z due to the plane tilt (Δz ≈ ±x_local) produces Δy = ty·Δz ≪ wire pitch for typical angles (ty ~ 2 mrad), so the correction is negligible.

### Analytic transmission values

| Element | Formula | T |
|---------|---------|---|
| grid1 (MN4 square mesh) | ((p − t) / p)² | **0.94897** |
| wire plane (single, 1-D) | (p − t) / p | **0.98000** |
| grid2 (MN8 square mesh) | ((p − t) / p)² | **0.89577** |

These are displayed in the Efficiencies sidebar without requiring a simulation run.

---

## Per-plane random offsets

Each plane receives an independent static offset (dx, dy) drawn from Uniform[−A, +A] at the start of each run. Default amplitude A = **0.5 mm**; adjustable 0–5 mm in the sidebar.

Offsets affect both the aperture test and the wire-hit blocking coordinate (via local coordinates x_local = x − dx, y_local = y − dy). A separate "offset seed" controls the offset RNG independently from the physics seed.

---

## Signals and counters

| Counter | Definition |
|---------|------------|
| N_generated | All simulated particles |
| N_pass_grid1 | Pass grid1 wire **and** aperture → **start signal** |
| N_reach_grid2 | Reach z_grid2 without aperture loss → **stop signal** |
| N_tof_defined | start AND stop |
| N_tof_recorded | tof_defined AND Bernoulli(η_MCP) |
| N_pass_grid2 | reach_grid2 AND pass grid2 wire check |
| N_reach_IC_geometric | pass_grid2 AND within IC aperture |
| N_detected_IC | reach_IC AND Bernoulli(η_IC) |
| **N_coin_TOF_IC** | **tof_recorded AND IC_detected** — primary measured sample |

**Stop signal rule**: wire hits at any wire plane do NOT terminate propagation. Only aperture losses (|x| or |y| > 25 mm) remove a particle from the geometric survival mask.

---

## TOF

TOF is measured between grid1 (start) and grid2 (stop):

```
TOF = Δz · √(1 + θx² + θy²) / v
```

A Gaussian timing resolution (default **400 ps FWHM**) is applied to recorded TOF events. Set to 0 for ideal timing.

---

## Headline rates

All rates are relative to N_generated:

| Rate | Formula |
|------|---------|
| TOF recorded fraction | N_tof_recorded / N_generated |
| IC detected fraction | N_detected_IC / N_generated |
| **Coincidence fraction** | **N_coin_TOF_IC / N_generated** |
| Grid transmission | N_pass_grid2 / N_generated |
| **Recovered MCP efficiency** | N_coin_TOF_IC / N_detected_IC |
| **Recovered IC efficiency** | N_coin_TOF_IC / (N_tof_recorded × T_grid2) |

**Recovered MCP efficiency**: fraction of IC-detected events that also produced a TOF signal. Estimates the combined MCP + start-plane efficiency as seen from the IC side.

**Recovered IC efficiency**: fraction of TOF events expected to reach the IC (after grid2 analytic transmission T_grid2) that were actually detected. Estimates η_IC × wire-plane transmission product as seen from the TOF side.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /config | Default SimParams + geometry JSON |
| POST | /simulate | Run MC → histograms, stats, scatter, heatmap, plane offsets |
| POST | /export | Stream event-level CSV (≤ 50 k rows) |

---

## File layout

```
backend/
  sim.py          Vectorised NumPy MC core
  main.py         FastAPI endpoints
  models.py       Pydantic v2 request/response models
  requirements.txt

frontend/
  index.html      Single-page app (Plotly.js, vanilla JS)

tests/
  test_sim.py     Unit tests for simulation invariants
```
