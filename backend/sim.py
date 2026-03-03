"""
Core Monte Carlo simulation — vectorised with NumPy.

Physics assumptions
───────────────────
1. Ballistic (straight-line) propagation:
       x(z) = x₀ + θx · (z − z_source)
       y(z) = y₀ + θy · (z − z_source)
   where θx = px/pz and θy = py/pz (small-angle approximation).

2. All particles are monoenergetic; energy spread is NOT modelled.

3. Relativistic kinematics (toggle):
       γ = 1 + T/(A·u·c²),   β = √(1 − 1/γ²),   v = β·c
   Non-relativistic fallback:
       v = √(2·T/m₀)

4. Path-length correction for TOF:
       L_eff = Δz · √(1 + θx² + θy²)
   so  TOF = L_eff / v.

5. Wire hit model (periodic pattern with a wire centred at local origin):
       u_mod = ((u + p/2) mod p) − p/2
       hit   ⟺   |u_mod| < t/2
   where p = pitch, t = wire thickness, u is the blocking coordinate.

6. Square apertures (5 cm × 5 cm) are enforced at every element in
   each plane's LOCAL coordinates (accounting for per-plane random offsets).

Geometry (all in metres)
─────────────────────────
MCP1:
  grid1  : z = 0.000–0.000032   MN4: 1238 µm pitch, 32 µm thick (z-extent = wire thickness), square mesh
  WP1    : z = 0.010   plane ⊥ z-axis; wires run along y (vertical); blocking coord u = x_local
  WP2    : z = 0.035   plane tilted +45° to z-axis; wires run along y; blocking coord u = x_local.
           2.5 cm downstream of WP1.  Wire length √50 cm = 5√2 cm; 5 cm wide in x.
  WP3    : z = 0.037   plane tilted +45° to z-axis; wires run along y; parallel to WP2.
           2 mm downstream of WP2.

MCP2 (upstream side of grid2, z_grid2 = 0.520 m):
  WP4    : z = z_grid2 - 0.039   plane tilted −45° to z-axis; wires run along y (u = x_local).
           First in beam order; parallel to WP5, 2 mm upstream. Wire length √50 cm; 5 cm wide in x.
  WP5    : z = z_grid2 - 0.037   plane tilted −45° to z-axis; wires run along y; 2.5 cm upstream of WP6.
  WP6    : z = z_grid2 - 0.012   plane ⊥ z-axis; wires run along y (vertical); blocking coord u = x_local
  grid2  : z = 0.520–0.520043   MN8: 803 µm pitch, 43 µm thick (z-extent = wire thickness), square mesh

IC plane : z = 0.620   (= z_grid2 + 0.10 m)

Key implementation notes
─────────────────────────
- STOP SIGNAL: generated when particle reaches z_grid2 within aperture,
  regardless of grid2 wire hit. TOF = (START AND STOP) × η_MCP.
  Wire hits at grid1–WP6 terminate propagation (remove from alive mask);
  grid2 wire hit only prevents the particle from continuing to the IC.
- COINCIDENCE: N_coin_TOF_IC = events with BOTH TOF recorded AND IC detected.
  All "measured yield" statistics use this.
- RANDOM OFFSETS: each plane gets a static (dx, dy) uniform in [-offset_amp, +offset_amp].
  Aperture test and wire-phase both use local coordinates: u_local = u - d_u.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT  = 299_792_458.0      # m/s
AMU_MEV  = 931.494_061_0      # MeV/u
AMU_KG   = 1.660_539_07e-27   # kg
MEV_TO_J = 1.602_176_634e-13  # J / MeV

# ── Beamline geometry ─────────────────────────────────────────────────────────
Z_GRID1  = 0.000   # m
Z_GRID2  = 0.520   # m   (52 cm downstream of grid1)
Z_IC     = 0.620   # m   (10 cm downstream of grid2)
APERTURE_HALF = 0.025   # m   → ±2.5 cm; 5 × 5 cm active area

# Wire planes as (z [m], blocking_coord, label)
# All wire planes have wires running along y (vertical) → blocking coordinate is x_local.
# WP2/WP3 are physically tilted +45° to z (in x-z plane); WP4/WP5 tilted −45°.
# The tilt shifts the crossing z by ≈ ±x_local, but for wire-hit purposes the
# correction Δx = tx·Δz ≪ wire pitch so the nominal-z approximation is used.
_WP_DEFS: List[Tuple[float, str, str]] = [
    (0.010,             'x', 'WP1'),  # MCP1 — ⊥ to z, vertical wires
    (0.035,             'x', 'WP2'),  # MCP1 — tilted +45° to z, vertical wires
    (0.037,             'x', 'WP3'),  # MCP1 — tilted +45° to z, vertical wires
    (Z_GRID2 - 0.039,   'x', 'WP4'),  # MCP2 — tilted −45° to z, vertical wires
    (Z_GRID2 - 0.037,   'x', 'WP5'),  # MCP2 — tilted −45° to z, vertical wires
    (Z_GRID2 - 0.012,   'x', 'WP6'),  # MCP2 — ⊥ to z, vertical wires
]

# Plane names in order (for offset table)
PLANE_NAMES = ['grid1', 'WP1', 'WP2', 'WP3', 'WP6', 'WP5', 'WP4', 'grid2', 'IC']
N_PLANES    = len(PLANE_NAMES)   # 9

# Grid MN4  (grid1) — square mesh
GRID1_PITCH = 1_238e-6   # m
GRID1_THICK =    32e-6   # m

# Wire planes — cylindrical wires, 20 µm diameter, 1 mm pitch
WP_PITCH = 1_000e-6   # m   pitch = 1 mm
WP_THICK =    20e-6   # m   wire diameter (cylindrical cross-section)

# Grid MN8  (grid2) — square mesh
GRID2_PITCH =  803e-6   # m
GRID2_THICK =   43e-6   # m

# z positions of each plane (same order as PLANE_NAMES)
PLANE_Z = [
    Z_GRID1,
    0.010,              # WP1
    0.035,              # WP2: 2.5 cm downstream of WP1
    0.037,              # WP3: 2 mm downstream of WP2
    Z_GRID2 - 0.012,    # WP6: last MCP2 plane before grid2
    Z_GRID2 - 0.037,    # WP5: 2.5 cm upstream of WP6
    Z_GRID2 - 0.039,    # WP4: 2 mm upstream of WP5
    Z_GRID2,
    Z_IC,
]


# ── Small helpers ─────────────────────────────────────────────────────────────

def fwhm_to_sigma(fwhm: float) -> float:
    """Convert Full-Width Half-Maximum to Gaussian σ."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def wire_hit(coord: np.ndarray, pitch: float, half_thick: float) -> np.ndarray:
    """
    Boolean mask: True where *coord* intersects a wire.

    Wires are centred at 0, ±pitch, ±2·pitch, …
    Periodic mapping: u_mod = ((u + pitch/2) mod pitch) − pitch/2
    Hit condition: |u_mod| < half_thick
    """
    u_mod = np.mod(coord + pitch * 0.5, pitch) - pitch * 0.5
    return np.abs(u_mod) < half_thick


def aperture_ok(x: np.ndarray, y: np.ndarray,
                half: float = APERTURE_HALF) -> np.ndarray:
    """True where (x, y) is within the square active aperture."""
    return (np.abs(x) <= half) & (np.abs(y) <= half)


def compute_kinematics(A: int, energy_mev_per_u: float,
                       relativistic: bool) -> Tuple[float, float, float]:
    """
    Return (v [m/s], β, γ) for a monoenergetic beam of ions.
    """
    E_kin_mev = energy_mev_per_u * A
    m0c2_mev  = A * AMU_MEV

    if relativistic:
        gamma = (E_kin_mev + m0c2_mev) / m0c2_mev
        beta2 = max(1.0 - 1.0 / (gamma * gamma), 0.0)
        beta  = np.sqrt(beta2)
        v     = beta * C_LIGHT
    else:
        E_kin_J = E_kin_mev * MEV_TO_J
        m0_kg   = A * AMU_KG
        v       = np.sqrt(2.0 * E_kin_J / m0_kg)
        beta    = v / C_LIGHT
        gamma   = 1.0 / np.sqrt(max(1.0 - beta * beta, 1e-30))

    return v, beta, gamma


def analytic_T(pitch: float, thick: float, axes: int = 1) -> float:
    """
    Analytic open-area fraction (geometric transmission).

    axes = 1 : single set of parallel wires  (wire plane)
    axes = 2 : square mesh (x-wires × y-wires)
    """
    t1 = (pitch - thick) / pitch
    return t1 ** axes


def generate_offsets(offset_amp_m: float, seed: Optional[int],
                     n_planes: int = N_PLANES) -> np.ndarray:
    """
    Generate static per-plane (dx, dy) offsets in metres.

    Returns array of shape (n_planes, 2).
    Uses a separate RNG seeded from `seed` so the physics RNG
    seed is not consumed by the offset draws.
    """
    rng_off = np.random.default_rng(seed)
    return rng_off.uniform(-offset_amp_m, offset_amp_m, size=(n_planes, 2))


# ── Main simulation ───────────────────────────────────────────────────────────

def run_simulation(
    N: int,
    D_source: float,
    fwhm_xy: float,
    fwhm_angle: float,
    energy_mev_per_u: float,
    A: int,
    Z: int,
    eta_MCP: float,
    eta_IC: float,
    n_bins: int,
    seed: Optional[int],
    relativistic: bool,
    fill_ic_detected: bool,
    offset_amp_m: float = 0.0005,      # 0.5 mm default
    offsets: Optional[np.ndarray] = None,  # shape (N_PLANES, 2), overrides amp+seed
    tof_fwhm_ps: float = 400.0,        # TOF timing resolution FWHM in picoseconds
) -> Tuple[Dict, Dict, Dict, Dict, np.ndarray]:
    """
    Generate N particles and propagate them through the beamline.

    Returns
    -------
    histograms : dict  keyed by plot name
    stats      : dict  summary counters and derived quantities
    scatter_ic : dict  x/y scatter data (downsampled)
    hist2d_ic  : dict  2D histogram data
    offsets_mm : ndarray shape (N_PLANES, 2)  per-plane offsets in mm
    """
    t_wall_start = time.perf_counter()

    # ── Plane offsets ─────────────────────────────────────────────────────────
    if offsets is None:
        offsets = generate_offsets(offset_amp_m, seed, N_PLANES)
    # offsets[i] = (dx, dy) in metres for plane i
    # Plane index mapping (matches PLANE_NAMES order):
    #   0=grid1, 1=WP1, 2=WP2, 3=WP3, 4=WP6, 5=WP5, 6=WP4, 7=grid2, 8=IC
    O_GRID1 = offsets[0]
    O_WP1   = offsets[1]
    O_WP2   = offsets[2]
    O_WP3   = offsets[3]
    O_WP6   = offsets[4]
    O_WP5   = offsets[5]
    O_WP4   = offsets[6]
    O_GRID2 = offsets[7]
    O_IC    = offsets[8]

    rng = np.random.default_rng(seed)

    sigma_xy    = fwhm_to_sigma(fwhm_xy)
    sigma_angle = fwhm_to_sigma(fwhm_angle)
    z_source    = -D_source

    # ── Generate phase space ──────────────────────────────────────────────────
    x0 = rng.normal(0.0, sigma_xy,    N).astype(np.float64)
    y0 = rng.normal(0.0, sigma_xy,    N).astype(np.float64)
    tx = rng.normal(0.0, sigma_angle, N).astype(np.float64)
    ty = rng.normal(0.0, sigma_angle, N).astype(np.float64)

    # ── Particle speed ────────────────────────────────────────────────────────
    v, beta, gamma = compute_kinematics(A, energy_mev_per_u, relativistic)

    # ── Position helper ───────────────────────────────────────────────────────
    def pos_at(z: float) -> Tuple[np.ndarray, np.ndarray]:
        dz = float(z - z_source)
        return x0 + tx * dz, y0 + ty * dz

    def local_xy(z: float, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Global position minus plane offset (local coordinates)."""
        xg, yg = pos_at(z)
        return xg - offset[0], yg - offset[1]

    # ── Geometric survival mask ───────────────────────────────────────────────
    # This mask tracks whether each particle is still propagating.
    # Both aperture losses AND wire hits remove a particle from alive.
    alive = np.ones(N, dtype=bool)   # all particles alive initially

    # ── Grid 1 (z = 0) ────────────────────────────────────────────────────────
    xl_g1, yl_g1 = local_xy(Z_GRID1, O_GRID1)
    ap_g1        = aperture_ok(xl_g1, yl_g1)
    hit_g1       = (wire_hit(xl_g1, GRID1_PITCH, GRID1_THICK * 0.5) |
                    wire_hit(yl_g1, GRID1_PITCH, GRID1_THICK * 0.5))
    pass_g1      = ap_g1 & ~hit_g1

    start_signal = pass_g1   # start-signal events (passed grid1 wire + aperture)

    # Aperture loss then wire absorption at grid1
    alive     = alive & ap_g1
    n_wire_g1 = int((alive & hit_g1).sum())
    alive     = alive & ~hit_g1

    # ── MCP1 wire planes ──────────────────────────────────────────────────────

    # WP1 at z = 0.010 m, wires along y (u = x_local)
    xl_wp1, yl_wp1 = local_xy(0.010, O_WP1)
    alive          = alive & aperture_ok(xl_wp1, yl_wp1)
    hit_wp1        = wire_hit(xl_wp1, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp1     = int((alive & hit_wp1).sum())
    alive          = alive & ~hit_wp1

    # WP2 at z = 0.035 m — plane tilted +45° to z-axis, wires along y (u = x_local)
    xl_wp2, yl_wp2 = local_xy(0.035, O_WP2)
    alive          = alive & aperture_ok(xl_wp2, yl_wp2)
    hit_wp2        = wire_hit(xl_wp2, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp2     = int((alive & hit_wp2).sum())
    alive          = alive & ~hit_wp2

    # WP3 at z = 0.037 m — plane tilted +45° to z-axis, wires along y; parallel to WP2
    xl_wp3, yl_wp3 = local_xy(0.037, O_WP3)
    alive          = alive & aperture_ok(xl_wp3, yl_wp3)
    hit_wp3        = wire_hit(xl_wp3, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp3     = int((alive & hit_wp3).sum())
    alive          = alive & ~hit_wp3

    # ── MCP2 wire planes (upstream of grid2, in beam order) ──────────────────

    # WP4 at z = z_grid2 - 0.039 — plane tilted −45° to z-axis, wires along y (u = x_local)
    z_wp4 = Z_GRID2 - 0.039
    xl_wp4, yl_wp4 = local_xy(z_wp4, O_WP4)
    alive          = alive & aperture_ok(xl_wp4, yl_wp4)
    hit_wp4        = wire_hit(xl_wp4, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp4     = int((alive & hit_wp4).sum())
    alive          = alive & ~hit_wp4

    # WP5 at z = z_grid2 - 0.037 — plane tilted −45° to z-axis, wires along y; parallel to WP4
    z_wp5 = Z_GRID2 - 0.037
    xl_wp5, yl_wp5 = local_xy(z_wp5, O_WP5)
    alive          = alive & aperture_ok(xl_wp5, yl_wp5)
    hit_wp5        = wire_hit(xl_wp5, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp5     = int((alive & hit_wp5).sum())
    alive          = alive & ~hit_wp5

    # WP6 at z = z_grid2 - 0.012, wires along y (u = x_local); last MCP2 plane before grid2
    z_wp6 = Z_GRID2 - 0.012
    xl_wp6, yl_wp6 = local_xy(z_wp6, O_WP6)
    alive          = alive & aperture_ok(xl_wp6, yl_wp6)
    hit_wp6        = wire_hit(xl_wp6, WP_PITCH, WP_THICK * 0.5)
    n_wire_wp6     = int((alive & hit_wp6).sum())
    alive          = alive & ~hit_wp6

    # ── Grid 2 — STOP SIGNAL position ─────────────────────────────────────────
    # Stop signal fires for any particle that reaches z_grid2 within aperture,
    # regardless of whether it hits a grid2 wire.  The wire hit only controls
    # whether the particle continues to the IC.
    xl_g2, yl_g2 = local_xy(Z_GRID2, O_GRID2)
    ap_g2        = aperture_ok(xl_g2, yl_g2)
    reach_g2     = alive & ap_g2
    stop_signal  = reach_g2.copy()     # stop fires on reaching grid2, wire-hit independent

    hit_g2    = (wire_hit(xl_g2, GRID2_PITCH, GRID2_THICK * 0.5) |
                 wire_hit(yl_g2, GRID2_PITCH, GRID2_THICK * 0.5))
    n_wire_g2 = int((reach_g2 & hit_g2).sum())
    pass_g2   = reach_g2 & ~hit_g2    # only mesh-passers continue to IC

    # tof_defined = start AND stop (both start_signal and reach_g2 are per-event)
    tof_defined = start_signal & stop_signal

    # ── TOF ───────────────────────────────────────────────────────────────────
    dz_tof   = Z_GRID2 - Z_GRID1      # = 0.520 m
    path_len = dz_tof * np.sqrt(1.0 + tx * tx + ty * ty)
    tof_ns_all = (path_len / v) * 1.0e9   # ns, for all N events

    tof_mcp_accept  = tof_defined & (rng.random(N) < eta_MCP)
    tof_ns_recorded = tof_ns_all[tof_mcp_accept]

    # ── TOF timing resolution (Gaussian smearing) ─────────────────────────────
    if tof_fwhm_ps > 0.0 and len(tof_ns_recorded) > 0:
        sigma_tof_ns    = fwhm_to_sigma(tof_fwhm_ps * 1e-3)   # ps → ns → sigma
        tof_ns_recorded = tof_ns_recorded + rng.normal(0.0, sigma_tof_ns,
                                                        len(tof_ns_recorded))

    # ── IC plane ──────────────────────────────────────────────────────────────
    # Particle must have passed grid2 wire check AND be within IC aperture
    xl_ic, yl_ic = local_xy(Z_IC, O_IC)
    ap_ic        = aperture_ok(xl_ic, yl_ic)
    reach_ic     = pass_g2 & ap_ic
    ic_det       = reach_ic & (rng.random(N) < eta_IC)

    # Histogram fill mask for IC position
    ic_mask = ic_det if fill_ic_detected else reach_ic

    # ── Coincidence ────────────────────────────────────────────────────────────
    # IMPORTANT: N_coin_TOF_IC is the primary "measured" sample.
    coin_tof_ic = tof_mcp_accept & ic_det

    # Legacy union (for the three-card panel kept from previous version)
    tof_or_ic = tof_mcp_accept | ic_det

    # ── Counters ──────────────────────────────────────────────────────────────
    n_gen        = N
    n_sg1        = int(start_signal.sum())
    n_rg2        = int(stop_signal.sum())   # geometrically reach grid2
    n_tof_def    = int(tof_defined.sum())
    n_tof_rec    = int(tof_mcp_accept.sum())
    n_pg2        = int(pass_g2.sum())
    n_ric        = int(reach_ic.sum())
    n_dic        = int(ic_det.sum())
    n_coin       = int(coin_tof_ic.sum())
    n_tof_or_ic  = int(tof_or_ic.sum())

    tof_mean = float(np.mean(tof_ns_recorded)) if len(tof_ns_recorded) else 0.0
    tof_rms  = float(np.std( tof_ns_recorded)) if len(tof_ns_recorded) else 0.0

    elapsed = time.perf_counter() - t_wall_start

    stats = dict(
        N_generated              = n_gen,
        N_pass_grid1             = n_sg1,
        N_reach_grid2            = n_rg2,
        N_tof_defined            = n_tof_def,
        N_tof_recorded           = n_tof_rec,
        N_pass_grid2             = n_pg2,
        N_reach_IC_geometric     = n_ric,
        N_detected_IC            = n_dic,
        N_coin_TOF_IC            = n_coin,

        frac_pass_grid1                = n_sg1     / n_gen    if n_gen    else 0.0,
        frac_reach_grid2_of_start      = n_rg2     / n_sg1    if n_sg1    else 0.0,
        frac_tof_recorded_of_defined   = n_tof_rec / n_tof_def if n_tof_def else 0.0,
        frac_IC_det_of_geometric       = n_dic     / n_ric     if n_ric    else 0.0,

        # ── Section-5 headline rates ───────────────────────────────────────────
        frac_tof_recorded      = n_tof_rec / n_gen if n_gen else 0.0,
        frac_IC_detected       = n_dic     / n_gen if n_gen else 0.0,
        frac_coincidence       = n_coin    / n_gen if n_gen else 0.0,

        # ── Legacy derived quantities ─────────────────────────────────────────
        N_tof_or_ic             = n_tof_or_ic,
        frac_IC_of_union        = n_dic     / n_tof_or_ic if n_tof_or_ic else 0.0,
        frac_TOF_of_union       = n_tof_rec / n_tof_or_ic if n_tof_or_ic else 0.0,
        transmission_both_grids = n_pg2     / n_gen       if n_gen       else 0.0,

        # ── Per-plane wire-hit counts (among particles alive at that plane) ──────
        N_wire_hit_grid1 = n_wire_g1,
        N_wire_hit_WP1   = n_wire_wp1,
        N_wire_hit_WP2   = n_wire_wp2,
        N_wire_hit_WP3   = n_wire_wp3,
        N_wire_hit_WP4   = n_wire_wp4,
        N_wire_hit_WP5   = n_wire_wp5,
        N_wire_hit_WP6   = n_wire_wp6,
        N_wire_hit_grid2 = n_wire_g2,

        tof_mean_ns        = tof_mean,
        tof_rms_ns         = tof_rms,
        velocity_m_per_s   = v,
        beta               = beta,
        gamma              = gamma,
        grid1_analytic_T   = analytic_T(GRID1_PITCH, GRID1_THICK, axes=2),
        wp_analytic_T_per_plane = analytic_T(WP_PITCH, WP_THICK, axes=1),
        grid2_analytic_T   = analytic_T(GRID2_PITCH, GRID2_THICK, axes=2),
        elapsed_s          = elapsed,
    )

    # ── Histograms ────────────────────────────────────────────────────────────
    def mkhist(data, xlabel: str, label: str) -> dict:
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) == 0:
            counts = np.zeros(n_bins, dtype=np.int64)
            edges  = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            counts, edges = np.histogram(arr, bins=n_bins)
        return dict(
            edges  = edges.tolist(),
            counts = counts.tolist(),
            label  = label,
            xlabel = xlabel,
            ylabel = "Counts",
            mean   = float(np.mean(arr)) if len(arr) else None,
            std    = float(np.std(arr))  if len(arr) else None,
            n      = int(len(arr)),
        )

    # x_local for IC (global pos), convert to mm
    x_ic_mm = (x0 + tx * (Z_IC - z_source))[ic_mask] * 1e3
    y_ic_mm = (y0 + ty * (Z_IC - z_source))[ic_mask] * 1e3

    histograms = {
        "x0":   mkhist(x0 * 1e3,          "x₀ (mm)",     "Initial x₀"),
        "y0":   mkhist(y0 * 1e3,          "y₀ (mm)",     "Initial y₀"),
        "tx":   mkhist(tx * 1e3,          "θx (mrad)",   "Initial θx"),
        "ty":   mkhist(ty * 1e3,          "θy (mrad)",   "Initial θy"),
        "x_ic": mkhist(x_ic_mm,           "x_IC (mm)",   "Final x at IC"),
        "y_ic": mkhist(y_ic_mm,           "y_IC (mm)",   "Final y at IC"),
        "tof":  mkhist(tof_ns_recorded,   "TOF (ns)",    "TOF_MCP"),
    }

    # ── 2-D position data (scatter + heatmap) ─────────────────────────────────
    xs_mm_full = x_ic_mm.copy()
    ys_mm_full = y_ic_mm.copy()
    n_ic_total = len(xs_mm_full)

    _BIN_MM    = 3.0
    _BIN_EDGES = np.arange(-30.0, 30.0 + _BIN_MM, _BIN_MM)
    if n_ic_total > 0:
        H, xe, ye = np.histogram2d(xs_mm_full, ys_mm_full,
                                   bins=[_BIN_EDGES, _BIN_EDGES])
    else:
        H  = np.zeros((len(_BIN_EDGES) - 1, len(_BIN_EDGES) - 1))
        xe = ye = _BIN_EDGES.copy()
    hist2d_ic = dict(
        x_edges     = xe.tolist(),
        y_edges     = ye.tolist(),
        counts      = H.T.tolist(),
        n_total     = n_ic_total,
        bin_size_mm = _BIN_MM,
    )

    _N_SCATTER_MAX = 5_000
    if n_ic_total > _N_SCATTER_MAX:
        idx     = rng.choice(n_ic_total, _N_SCATTER_MAX, replace=False)
        xs_sc   = xs_mm_full[idx]
        ys_sc   = ys_mm_full[idx]
        n_shown = _N_SCATTER_MAX
    else:
        xs_sc   = xs_mm_full
        ys_sc   = ys_mm_full
        n_shown = n_ic_total
    scatter_ic = dict(
        x       = xs_sc.tolist(),
        y       = ys_sc.tolist(),
        n_total = n_ic_total,
        n_shown = int(n_shown),
    )

    return histograms, stats, scatter_ic, hist2d_ic, offsets
