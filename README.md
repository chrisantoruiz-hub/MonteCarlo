# Monte Carlo Beamline Simulator v2

Browser-based Monte Carlo simulator for particle trajectories through MCP foils/grids
and wire planes. FastAPI backend + vanilla-JS + Plotly.js frontend.

---

## Quick start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000

# In a separate terminal (or just open the file):
open ../frontend/index.html
```

---

## Geometry

### MCP1 (z = 0 – 0.012 m)

| Element | z (m)  | Description |
|---------|--------|-------------|
| grid1   | 0.000  | MN4 square mesh: pitch 1238 µm, t = 32 µm (x-wires + y-wires) |
| WP1     | 0.010  | Wire plane: vertical wires, blocking coord u = x_local |
| WP2     | 0.010  | Wire plane: **+45° wires**, blocking coord u = (x+y)/√2, same z as WP1 |
| WP3     | 0.012  | Wire plane: +45° wires (same orientation as WP2), 2 mm downstream |

### MCP2 (z = 0.508 – 0.520 m, mirror of MCP1)

| Element | z (m)  | Description |
|---------|--------|-------------|
| WP6     | 0.508  | +45° wires (mirror of WP3) |
| WP5     | 0.510  | +45° wires (mirror of WP2), same z as WP4 |
| WP4     | 0.510  | Vertical wires (mirror of WP1), same z as WP5 |
| grid2   | 0.520  | MN8 square mesh: pitch 803 µm, t = 43 µm (x-wires + y-wires) |

### Downstream

| Element | z (m)  | Description |
|---------|--------|-------------|
| IC      | 0.620  | Ionisation chamber, η_IC efficiency, same ±2.5 cm aperture |

### Wire plane geometry (all 6 wire planes)

- Pitch = 2500 µm, thickness = 50 µm, T = 98 % per plane (1D)
- All wire planes are **perpendicular to the beam axis**
- WP2/WP3/WP5/WP6: wires are physically oriented at 45° in the x–y plane;
  blocking coordinate is `u = x_local·cos(45°) + y_local·sin(45°)`

### 45-degree wire plane interpretation

The spec says WP2 is "at 45 degrees to WP1". This is interpreted as a
**wire-orientation rotation** in the x–y plane, NOT a tilt of the physical plane
relative to the beam. The plane itself remains perpendicular to the beam.
This is the only self-consistent interpretation in a straight-line MC
(tilted planes would require path-length corrections and are not implied by the
experimental description). All five 45°-planes (WP2, WP3, WP4, WP5, WP6) use
the same blocking coordinate `u = (x+y)/√2` in each plane's local frame.

---

## Random per-plane offsets

At the start of each run, each physical plane receives an independent static
offset `(dx, dy)` drawn from `Uniform[-amplitude, +amplitude]` in both x and y.
The default amplitude is **0.5 mm**; the slider can set it 0–5 mm.

These offsets affect:
1. **Aperture test**: particle is lost if `|x - dx| > 25 mm` or `|y - dy| > 25 mm`
2. **Wire-hit test**: blocking coordinate uses `x_local = x - dx` (or `y - dy`)

The offsets table below the histograms shows the actual values used in each run.
A separate "offset seed" controls the offset RNG independently of the physics seed.

---

## Signals and counters

| Counter | Definition |
|---------|-----------|
| N_generated | All simulated particles |
| N_pass_grid1 | Pass grid1 wire + aperture → **start signal** |
| N_reach_grid2 | Reach z_grid2 without aperture loss → **stop signal** |
| N_tof_defined | start AND stop |
| N_tof_recorded | tof_defined AND Bernoulli(η_MCP) |
| N_pass_grid2 | reach_grid2 AND pass grid2 wire check |
| N_reach_IC_geometric | pass_grid2 AND within IC aperture |
| N_detected_IC | reach_IC AND Bernoulli(η_IC) |
| **N_coin_TOF_IC** | **tof_recorded AND IC_detected** — primary measured sample |

**Stop signal rule**: wire hits at any wire plane (WP1–WP6, grid2) do NOT
terminate propagation. Only aperture losses (|x| > 25 mm or |y| > 25 mm)
remove a particle from the geometric survival mask. A particle that aperture-
survives to z_grid2 generates a stop signal regardless of grid2 wire hit.

---

## Headline rates (bottom-of-page panel)

All rates are relative to N_generated:

| Rate | Formula |
|------|---------|
| TOF recorded fraction | N_tof_recorded / N_generated |
| IC detected fraction | N_detected_IC / N_generated |
| **Coincidence fraction** | **N_coin_TOF_IC / N_generated** |

The coincidence is the "measured yield" — the only sample available to an
experimenter who requires both TOF and IC in every event.

---

## Validation

| Test | Expected behaviour |
|------|--------------------|
| η_MCP = 0 | N_tof_recorded = 0, N_coin = 0 |
| η_IC = 0 | N_detected_IC = 0, N_coin = 0 |
| offset_amp = 0, small angles | Near-maximum grid transmission (~94.9 % for grid1) |
| Large angle FWHM | Geometric losses increase, IC yield falls |

---

## File layout

```
backend/
  sim.py        Vectorised NumPy MC core
  main.py       FastAPI endpoints (/config, /simulate, /export)
  models.py     Pydantic v2 request/response models
  requirements.txt

frontend/
  index.html    Single-page app (Plotly.js, vanilla JS)

README.md
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /config | Default params + geometry JSON |
| POST | /simulate | Run MC, return 7 histograms + stats + scatter + heatmap + offsets |
| POST | /export | Stream event-level CSV (≤ 50 k rows) |
