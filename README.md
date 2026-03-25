# MCP and IC Efficiency Simulator

Browser-based Monte Carlo simulator for particle trajectories through a two-MCP beamline with grids, tilted wire planes, and an ionisation chamber. Determines MCP and IC efficiencies from coincidence spectra. FastAPI backend + vanilla JS + Plotly.js + MathJax frontend.

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

| Element | z (m)             | Description |
|---------|-------------------|-------------|
| grid1   | 0.000 – 0.000032  | MN4 square mesh: pitch 1238 µm, t = 32 µm (x- and y-wires); z-extent = wire thickness |
| WP1     | 0.010             | Plane **⊥ to beam**; wires along y (vertical; blocking coord u = x_local) |
| WP2     | 0.035             | Plane **tilted +45° to z-axis** (in x-z plane); wires along y (u = x_local) |
| WP3     | 0.037             | Plane **tilted +45° to z-axis**; wires along y; parallel to WP2, 2 mm downstream |

### MCP2 (z = 0.481 – 0.520 m)

| Element | z (m)             | Description |
|---------|-------------------|-------------|
| WP4     | 0.481             | Plane **tilted −45° to z-axis** (in x-z plane); wires along y (u = x_local); first in beam order |
| WP5     | 0.483             | Plane **tilted −45° to z-axis**; wires along y; parallel to WP4, 2 mm downstream |
| WP6     | 0.508             | Plane **⊥ to beam**; wires along y (vertical; blocking coord u = x_local) |
| grid2   | 0.520 – variable  | Square mesh; type selectable (see below); x- and y-wires; z-extent = wire thickness |

### Downstream

| Element | z (m)  | Description |
|---------|--------|-------------|
| IC      | 0.620  | Ionisation chamber, efficiency η_IC, same ±2.5 cm aperture |

### Wire plane common parameters

- Pitch = 1000 µm, wire diameter = 20 µm (cylindrical) → T = 98 % per plane (1-D analytic)
- All wires run along **y** (vertical; blocking coord u = x_local). The ±45° refers to the tilt of the **physical plane** relative to the beam axis, not the wire orientation.
- For the wire-hit check, the nominal-z approximation is used: the x-dependent shift in crossing-z due to the plane tilt (Δz ≈ ±x_local) produces Δx = tx·Δz ≪ wire pitch for typical angles (tx ~ 2 mrad), so the correction is negligible.

### Grid2 selectable mesh types

| Name | Pitch (µm) | Wire thickness (µm) | T_analytic |
|------|-----------|---------------------|------------|
| MN4  | 1238      | 32                  | 0.94897    |
| MN8  | 803       | 43                  | **0.89577** (default, standard geometry) |
| MN9  | 785       | 61                  | 0.85449    |
| MN14 | 440       | 68                  | 0.71735    |

T_analytic = ((p − t) / p)². The active type is selected in the Efficiencies sidebar; the geometry diagram and analytic-T display update immediately, and the selected pitch/thickness are forwarded to the simulation.

In the **alternative geometry** the same selector applies to all four MCP2 planes (grid2 + WP6 + WP5 + WP4); default switches to MN4.

### Analytic transmission values

| Element | Formula | T |
|---------|---------|---|
| grid1 (MN4 square mesh) | ((p − t) / p)² | **0.94897** |
| wire plane (single, 1-D) | (p − t) / p | **0.98000** |
| grid2 (type-dependent)   | ((p − t) / p)² | see table above |

These are displayed in the Efficiencies sidebar without requiring a simulation run.

---

## Alternative geometry (MCP2 reversed)

Toggled with the **Alt. Geometry** button in the header. MCP2 is reflected in z so that grid2 faces the beam first, followed by WP6, WP5, WP4 downstream.

### MCP2 — alternative order (z = 0.520 – 0.559 m)

| Element | z (m) | Description |
|---------|-------|-------------|
| grid2   | 0.520 | Square mesh; same selectable MN type applied to all MCP2 planes; **default MN4** |
| WP6     | 0.532 | Square mesh (same type as grid2); plane ⊥ to beam |
| WP5     | 0.557 | Square mesh; plane tilted **+45° to z-axis** (reflected from standard −45°) |
| WP4     | 0.559 | Square mesh; plane tilted **+45° to z-axis**; parallel to WP5, 2 mm downstream |

MCP1 (grid1 + WP1/WP2/WP3) is unchanged.

### Counting equations (alternative geometry)

```
N_MCP     = N_tot · τ_grid1 · τ_wires1 · τ_grid2 · η_MCP
N_IC      = N_tot · τ_grid1 · τ_wires1 · τ_grid2 · τ_wires2 · η_IC
N_MCP∩IC  = N_tot · τ_grid1 · τ_wires1 · τ_grid2 · τ_wires2 · η_MCP · η_IC
```

where τ_wires2 = T_MN³ (three square mesh planes with the selected MN type).

### Efficiency extraction (alternative geometry)

```
η_MCP = N_MCP∩IC / N_IC
η_IC  = N_MCP∩IC / (N_MCP · τ_wires2)
```

All upstream factors (N_tot, τ_grid1, τ_wires1, τ_grid2) cancel in both ratios.

---

## Per-plane random offsets

Each plane receives an independent static offset (dx, dy) drawn from Uniform[−A, +A] at the start of each run. Default amplitude A = **0.5 mm**; adjustable 0–5 mm in the sidebar.

Offsets affect both the aperture test and the wire-hit blocking coordinate (via local coordinates x_local = x − dx, y_local = y − dy). A separate "offset seed" controls the offset RNG independently from the physics seed.

---

## Propagation rules

A particle is removed from the **alive** mask (stops propagating) when it either:
- exits the ±25 mm aperture at any element, or
- hits a wire at **grid1 or WP1–WP6**.

Grid2 wire hits do **not** kill the alive mask — they only prevent the particle from continuing to the IC.

> **Alternative geometry**: grid2 wire hits **do** kill the alive mask (STOP requires passing the wire check), and WP6/WP5/WP4 wire hits kill the mask downstream of grid2.

---

## Signals and counters

| Counter | Definition |
|---------|------------|
| N_generated | All simulated particles |
| N_pass_grid1 | Alive at grid1, within aperture, **and** miss grid1 wires → **start signal** |
| N_reach_grid2 | Survive all upstream aperture + wire-hit losses and reach z_grid2 aperture → **stop signal** |
| N_tof_defined | start AND stop |
| N_tof_recorded | tof_defined AND Bernoulli(η_MCP) |
| N_pass_grid2 | reach_grid2 AND pass grid2 wire mesh |
| N_reach_IC_geometric | pass_grid2 AND within IC aperture |
| N_detected_IC | reach_IC AND Bernoulli(η_IC) |
| **N_coin_TOF_IC** | **tof_recorded AND IC_detected** — primary measured sample |
| N_wire_hit_X | Events alive at plane X that struck a wire (X = grid1, WP1–WP6, grid2) |

**Stop signal rule (standard)**: The stop signal fires whenever a particle reaches z_grid2 within aperture — the grid2 wire hit is irrelevant to TOF generation.

**Stop signal rule (alternative geometry)**: The stop signal fires only when a particle **passes through** grid2 wires (τ_grid2 enters the MCP counting equation).

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

### Standard geometry

| Rate | Formula |
|------|---------|
| TOF recorded fraction | N_tof_recorded / N_generated |
| IC detected fraction | N_detected_IC / N_generated |
| **Coincidence fraction** | **N_coin_TOF_IC / N_generated** |
| Grid transmission | N_pass_grid2 / N_generated |
| **Recovered MCP efficiency** | N_coin_TOF_IC / N_detected_IC |
| **Recovered IC efficiency** | N_coin_TOF_IC / (N_tof_recorded × T_grid2) |

### Alternative geometry

| Rate | Formula |
|------|---------|
| TOF recorded fraction | N_tof_recorded / N_generated |
| IC detected fraction | N_detected_IC / N_generated |
| **Coincidence fraction** | **N_coin_TOF_IC / N_generated** |
| Grid transmission | N_reach_IC_geometric / N_generated |
| **Recovered MCP efficiency** | N_coin_TOF_IC / N_detected_IC |
| **Recovered IC efficiency** | N_coin_TOF_IC / (N_tof_recorded × τ_wires2) |

where τ_wires2 = T_mesh³ (three MN mesh planes: WP6 + WP5 + WP4).

**Recovered MCP efficiency**: same formula in both geometries — fraction of IC-detected events that also produced a TOF signal.

**Recovered IC efficiency (standard)**: uses T_grid2 (single mesh plane) as the known geometric factor.

**Recovered IC efficiency (alternative)**: uses τ_wires2 = T_mesh³ as the known geometric factor. Note: a ~5% overestimate arises from spatial correlations — particles selected by the grid2 wire pattern land preferentially in the open areas of the downstream WP6/WP5/WP4 planes (same pitch, short spacing), making the effective τ_wires2 higher than the analytic T_mesh³ for a uniform beam.

The derivation of these efficiency estimators — including counting equations, cancellation of upstream transmissions, and Poisson uncertainties — is reproduced in the **Background Info** tab of the frontend (separate derivation for each geometry).

---

## Frontend tabs

### Simulation tab
Contains the full interactive simulation interface: parameter sidebar, run button, histograms, 2-D scatter / heatmap, derived-quantities panel, per-plane offset table, and geometry diagrams. The **Alt. Geometry** toggle in the header switches between standard and alternative MCP2 configurations; the beamline schematic, geometry z-view diagrams, mesh-type selector, analytic-T display, and derived-quantities formulas all update accordingly.

### Background Info tab
Renders a derivation document (via MathJax). The document shown depends on the active geometry mode:

**Standard geometry**: counting equations, cancellation of upstream transmissions, extraction formulas for η_MCP and η_IC, numerical example, Poisson uncertainty propagation.

**Alternative geometry**: modified counting equations with τ_grid2 in the MCP rate, derivation of η_IC = N_coin / (N_MCP · τ_wires2), numerical example with MN4 mesh, Poisson uncertainties.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /config | Default SimParams + geometry JSON (lists all grid2 types) |
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
  index.html      Single-page app (Plotly.js, MathJax, vanilla JS)

tests/
  test_sim.py     Unit tests for simulation invariants
```
