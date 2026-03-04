"""
Pydantic v2 request / response models for the Monte Carlo beamline simulator.

Unit conventions:
  - positions  : metres (m) in request; mm in histogram output
  - angles     : radians (rad) in request; mrad in histogram output
  - energy     : MeV/u
  - time       : nanoseconds (ns) in histogram output
  - offsets    : metres (m) internally; mm in the offsets table response
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class SimParams(BaseModel):
    """All tunable simulation parameters sent from the browser."""

    # Statistics
    N: int = Field(
        default=200_000, ge=10_000, le=2_000_000,
        description="Number of Monte Carlo events to generate",
    )

    # Beam source
    D_source: float = Field(
        default=0.50, ge=0.01, le=5.0,
        description="Distance of source plane upstream of grid1 (m)",
    )
    fwhm_xy: float = Field(
        default=0.002, ge=1e-4, le=0.10,
        description="Initial transverse spot FWHM (m); Gaussian in x and y",
    )
    fwhm_angle: float = Field(
        default=0.002, ge=1e-5, le=0.05,
        description="Initial angular-divergence FWHM (rad); Gaussian in θx and θy",
    )

    # Beam species / kinematics
    energy_mev_per_u: float = Field(
        default=1.0, ge=0.01, le=1_000.0,
        description="Kinetic energy per nucleon (MeV/u)",
    )
    A: int = Field(default=12, ge=1, le=250, description="Mass number (e.g. 12 for 12C)")
    Z: int = Field(default=6,  ge=1, le=120, description="Atomic number (e.g. 6 for carbon)")

    # Efficiencies
    eta_MCP: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="MCP detection efficiency applied to valid TOF events",
    )
    eta_IC: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Ionisation-chamber detection efficiency",
    )

    # Histogram / display
    n_bins: int = Field(
        default=100, ge=10, le=500,
        description="Number of bins in each histogram",
    )

    # Random plane offsets
    offset_amp_mm: float = Field(
        default=0.5, ge=0.0, le=5.0,
        description="Half-amplitude of uniform random per-plane offset in x and y (mm)",
    )
    offset_seed: Optional[int] = Field(
        default=None,
        description="Seed for the offset RNG (None = random). Separate from physics seed.",
    )

    # TOF timing resolution
    tof_fwhm_ps: float = Field(
        default=400.0, ge=0.0, le=2000.0,
        description="Gaussian timing resolution of TOF_MCP (FWHM in ps; 0 = ideal)",
    )

    # Grid2 mesh type (selectable)
    grid2_pitch_um: float = Field(
        default=803.0, ge=100.0, le=5000.0,
        description="Grid2 mesh pitch (µm); default = MN8",
    )
    grid2_thick_um: float = Field(
        default=43.0, ge=1.0, le=200.0,
        description="Grid2 wire thickness (µm); default = MN8",
    )

    # Optional / advanced
    seed: Optional[int] = Field(
        default=None,
        description="NumPy random seed for reproducibility (None = random)",
    )
    relativistic: bool = Field(
        default=True,
        description="Use relativistic kinematics when computing particle velocity",
    )
    fill_ic_detected: bool = Field(
        default=True,
        description=(
            "IC position histograms: True = only η_IC-accepted events, "
            "False = all events that reach IC geometrically"
        ),
    )


class ExportRequest(BaseModel):
    """Request body for /export endpoint."""
    N_export: int = Field(default=50_000, ge=100, le=50_000)
    sim_params: SimParams


# ── Histogram ────────────────────────────────────────────────────────────────

class HistogramData(BaseModel):
    """Pre-binned histogram returned by the backend."""
    edges: List[float]
    counts: List[int]
    label: str
    xlabel: str
    ylabel: str = "Counts"
    mean: Optional[float] = None
    std: Optional[float] = None
    n: Optional[int] = None


# ── Summary statistics ────────────────────────────────────────────────────────

class SimStats(BaseModel):
    # Raw counters
    N_generated: int
    N_pass_grid1: int
    N_reach_grid2: int
    N_tof_defined: int
    N_tof_recorded: int
    N_pass_grid2: int
    N_reach_IC_geometric: int
    N_detected_IC: int
    N_coin_TOF_IC: int           # COINCIDENCE: both TOF recorded AND IC detected

    # Derived efficiencies / fractions (chain)
    frac_pass_grid1: float
    frac_reach_grid2_of_start: float
    frac_tof_recorded_of_defined: float
    frac_IC_det_of_geometric: float

    # Section-5 headline rates (all relative to N_generated)
    frac_tof_recorded: float     # N_tof_recorded / N_generated
    frac_IC_detected: float      # N_detected_IC  / N_generated
    frac_coincidence: float      # N_coin_TOF_IC  / N_generated

    # Per-plane wire-hit counts (among particles geometrically alive at each plane)
    N_wire_hit_grid1: int
    N_wire_hit_WP1: int
    N_wire_hit_WP2: int
    N_wire_hit_WP3: int
    N_wire_hit_WP4: int
    N_wire_hit_WP5: int
    N_wire_hit_WP6: int
    N_wire_hit_grid2: int

    # TOF
    tof_mean_ns: float
    tof_rms_ns: float

    # Kinematics
    velocity_m_per_s: float
    beta: float
    gamma: float

    # Analytic open-area fractions
    grid1_analytic_T: float
    wp_analytic_T_per_plane: float
    grid2_analytic_T: float

    elapsed_s: float

    # Legacy union stats (kept for three-card panel)
    N_tof_or_ic: int
    frac_IC_of_union: float
    frac_TOF_of_union: float
    transmission_both_grids: float


# ── Plane offsets table ───────────────────────────────────────────────────────

class PlaneOffset(BaseModel):
    name: str
    z_m: float
    dx_mm: float
    dy_mm: float


# ── 2-D scatter data ──────────────────────────────────────────────────────────

class Scatter2D(BaseModel):
    x: List[float]
    y: List[float]
    n_total: int
    n_shown: int


# ── 2-D histogram (heatmap) ───────────────────────────────────────────────────

class Hist2D(BaseModel):
    x_edges: List[float]
    y_edges: List[float]
    counts: List[List[int]]
    n_total: int
    bin_size_mm: float


# ── Top-level response ────────────────────────────────────────────────────────

class SimResult(BaseModel):
    histograms: Dict[str, HistogramData]
    stats: SimStats
    scatter_ic: Scatter2D
    hist2d_ic: Hist2D
    plane_offsets: List[PlaneOffset]


# ── Config endpoint ───────────────────────────────────────────────────────────

class DefaultConfig(BaseModel):
    params: SimParams
    geometry: Dict[str, Any]
