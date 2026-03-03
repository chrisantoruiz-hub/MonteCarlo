"""
FastAPI backend for the Monte Carlo beamline simulator.

Endpoints
─────────
GET  /config      → default SimParams + geometry description
POST /simulate    → run simulation, return histogram data + summary stats
POST /export      → return CSV of event-level data (capped at 50 k events)

Start with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import (
    DefaultConfig,
    ExportRequest,
    Hist2D,
    HistogramData,
    PlaneOffset,
    Scatter2D,
    SimParams,
    SimResult,
    SimStats,
)
from sim import (
    APERTURE_HALF,
    GRID1_PITCH, GRID1_THICK,
    GRID2_PITCH, GRID2_THICK,
    WP_PITCH, WP_THICK,
    Z_GRID1, Z_GRID2, Z_IC,
    PLANE_NAMES, PLANE_Z,
    analytic_T,
    compute_kinematics,
    fwhm_to_sigma,
    generate_offsets,
    run_simulation,
)

app = FastAPI(
    title="Monte Carlo Beamline Simulator",
    version="2.0.0",
    description=(
        "Vectorised NumPy MC simulation of particle trajectories through "
        "a beamline with grids, wire planes, and a detection IC. "
        "New in v2: 45° wire planes, per-plane random offsets, coincidence tracking."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /config ───────────────────────────────────────────────────────────────────

@app.get("/config", response_model=DefaultConfig)
def get_config() -> DefaultConfig:
    return DefaultConfig(
        params=SimParams(),
        geometry={
            "aperture_cm": APERTURE_HALF * 200,
            "MCP1": {
                "grid1": {
                    "z_m":        Z_GRID1,
                    "z_extent_m": (Z_GRID1, round(Z_GRID1 + GRID1_THICK, 9)),
                    "type":       "MN4 (square mesh)",
                    "pitch_um":   round(GRID1_PITCH * 1e6, 1),
                    "thick_um":   round(GRID1_THICK * 1e6, 1),
                    "note":       "z-extent equals wire thickness",
                    "T_analytic": round(analytic_T(GRID1_PITCH, GRID1_THICK, axes=2), 5),
                    "wires":      "x and y",
                },
                "WP1": {"z_m": 0.010, "orientation": "horizontal wires (run along x; u = y_local); plane ⊥ z-axis"},
                "WP2": {"z_m": 0.035, "orientation": "horizontal wires (u = y_local); plane tilted +45° to z-axis", "note": "2.5 cm downstream of WP1; wire length √50 cm = 5√2 cm, 5 cm wide in x"},
                "WP3": {"z_m": 0.037, "orientation": "horizontal wires (u = y_local); plane tilted +45° to z-axis", "note": "parallel to WP2; 2 mm downstream"},
            },
            "MCP2": {
                "WP4": {"z_m": Z_GRID2 - 0.039, "orientation": "horizontal wires (u = y_local); plane tilted −45° to z-axis", "note": "first in beam order; parallel to WP5, 2 mm upstream; wire length √50 cm, 5 cm wide in x"},
                "WP5": {"z_m": Z_GRID2 - 0.037, "orientation": "horizontal wires (u = y_local); plane tilted −45° to z-axis", "note": "2.5 cm upstream of WP6"},
                "WP6": {"z_m": Z_GRID2 - 0.012, "orientation": "horizontal wires (run along x; u = y_local); plane ⊥ z-axis"},
                "grid2": {
                    "z_m":        Z_GRID2,
                    "z_extent_m": (Z_GRID2, round(Z_GRID2 + GRID2_THICK, 9)),
                    "type":       "MN8 (square mesh)",
                    "pitch_um":   round(GRID2_PITCH * 1e6, 1),
                    "thick_um":   round(GRID2_THICK * 1e6, 1),
                    "note":       "z-extent equals wire thickness",
                    "T_analytic": round(analytic_T(GRID2_PITCH, GRID2_THICK, axes=2), 5),
                    "wires":      "x and y",
                },
            },
            "ic_plane": {
                "z_m":  Z_IC,
                "note": "10 cm downstream of grid2; same 5×5 cm aperture",
            },
            "wire_planes_common": {
                "pitch_um": round(WP_PITCH * 1e6, 1),
                "thick_um": round(WP_THICK * 1e6, 1),
                "T_analytic_per_plane": round(analytic_T(WP_PITCH, WP_THICK, axes=1), 4),
                "note_orientations": (
                    "ALL wire planes have horizontal wires (run along x; blocking u = y_local). "
                    "WP1/WP6 planes are perpendicular to the beam axis. "
                    "WP2/WP3 planes are tilted +45° to the z-axis (in x-z plane). "
                    "WP4/WP5 planes are tilted −45° to the z-axis (in x-z plane). "
                    "Tilt shifts the crossing z by ≈ ±x_local but Δy = ty·Δz ≪ wire pitch so "
                    "the nominal-z approximation is used for wire-hit evaluation."
                ),
            },
        },
    )


# ── /simulate ─────────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimResult)
def simulate(params: SimParams) -> SimResult:
    """Run Monte Carlo simulation and return histogram data plus summary stats."""
    try:
        hists_raw, stats_raw, scatter_raw, hist2d_raw, offsets = run_simulation(
            N                = params.N,
            D_source         = params.D_source,
            fwhm_xy          = params.fwhm_xy,
            fwhm_angle       = params.fwhm_angle,
            energy_mev_per_u = params.energy_mev_per_u,
            A                = params.A,
            Z                = params.Z,
            eta_MCP          = params.eta_MCP,
            eta_IC           = params.eta_IC,
            n_bins           = params.n_bins,
            seed             = params.seed,
            relativistic     = params.relativistic,
            fill_ic_detected = params.fill_ic_detected,
            offset_amp_m     = params.offset_amp_mm * 1e-3,
            offsets          = None,
            tof_fwhm_ps      = params.tof_fwhm_ps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    histograms = {k: HistogramData(**v) for k, v in hists_raw.items()}
    stats      = SimStats(**stats_raw)
    scatter_ic = Scatter2D(**scatter_raw)
    hist2d_ic  = Hist2D(**hist2d_raw)

    plane_offsets = [
        PlaneOffset(
            name  = PLANE_NAMES[i],
            z_m   = PLANE_Z[i],
            dx_mm = round(float(offsets[i, 0]) * 1e3, 4),
            dy_mm = round(float(offsets[i, 1]) * 1e3, 4),
        )
        for i in range(len(PLANE_NAMES))
    ]

    return SimResult(
        histograms    = histograms,
        stats         = stats,
        scatter_ic    = scatter_ic,
        hist2d_ic     = hist2d_ic,
        plane_offsets = plane_offsets,
    )


# ── /export ───────────────────────────────────────────────────────────────────

@app.post("/export")
def export(req: ExportRequest) -> StreamingResponse:
    """Return a CSV file with event-level data for up to 50 000 events."""
    sp       = req.sim_params
    N_export = min(req.N_export, 50_000)

    try:
        hists_raw, stats_raw, _scatter, _hist2d, _offsets = run_simulation(
            N                = N_export,
            D_source         = sp.D_source,
            fwhm_xy          = sp.fwhm_xy,
            fwhm_angle       = sp.fwhm_angle,
            energy_mev_per_u = sp.energy_mev_per_u,
            A                = sp.A,
            Z                = sp.Z,
            eta_MCP          = sp.eta_MCP,
            eta_IC           = sp.eta_IC,
            n_bins           = 10,
            seed             = sp.seed,
            relativistic     = sp.relativistic,
            fill_ic_detected = sp.fill_ic_detected,
            offset_amp_m     = sp.offset_amp_mm * 1e-3,
            tof_fwhm_ps      = sp.tof_fwhm_ps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    import sim as _sim
    rng         = np.random.default_rng(sp.seed)
    sigma_xy    = _sim.fwhm_to_sigma(sp.fwhm_xy)
    sigma_angle = _sim.fwhm_to_sigma(sp.fwhm_angle)
    z_source    = -sp.D_source

    x0 = rng.normal(0.0, sigma_xy,    N_export)
    y0 = rng.normal(0.0, sigma_xy,    N_export)
    tx = rng.normal(0.0, sigma_angle, N_export)
    ty = rng.normal(0.0, sigma_angle, N_export)

    def pos_at(z):
        dz = z - z_source
        return x0 + tx * dz, y0 + ty * dz

    xic, yic = pos_at(_sim.Z_IC)
    v, _, _  = _sim.compute_kinematics(sp.A, sp.energy_mev_per_u, sp.relativistic)
    tof_ns   = (_sim.Z_GRID2 - _sim.Z_GRID1) * np.sqrt(1 + tx**2 + ty**2) / v * 1e9

    buf = io.StringIO()
    buf.write("x0_mm,y0_mm,tx_mrad,ty_mrad,xIC_mm,yIC_mm,tof_ns\n")
    for i in range(N_export):
        buf.write(
            f"{x0[i]*1e3:.4f},{y0[i]*1e3:.4f},"
            f"{tx[i]*1e3:.4f},{ty[i]*1e3:.4f},"
            f"{xic[i]*1e3:.4f},{yic[i]*1e3:.4f},"
            f"{tof_ns[i]:.6f}\n"
        )

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mc_export.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
