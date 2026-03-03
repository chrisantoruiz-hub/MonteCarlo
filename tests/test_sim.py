"""
Basic validation tests for the Monte Carlo simulation.

Run with:
    cd /path/to/MonteCarlo
    pip install pytest numpy
    pytest tests/test_sim.py -v
"""

import sys
import os
import math

import numpy as np
import pytest

# Make backend importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sim import (
    C_LIGHT, AMU_MEV, Z_GRID1, Z_GRID2, Z_IC,
    GRID1_PITCH, GRID1_THICK, WP_PITCH, WP_THICK,
    GRID2_PITCH, GRID2_THICK, APERTURE_HALF,
    fwhm_to_sigma, wire_hit, aperture_ok,
    compute_kinematics, analytic_T, run_simulation,
)


# ── Unit helpers ──────────────────────────────────────────────────────────────

class TestFwhmSigma:
    def test_round_trip(self):
        fwhm = 3.5
        sigma = fwhm_to_sigma(fwhm)
        # Gaussian at ±FWHM/2 should be at half-maximum
        assert abs(np.exp(-0.5 * (fwhm / 2 / sigma) ** 2) - 0.5) < 1e-9

    def test_known_value(self):
        # sigma = FWHM / 2.3548...
        sigma = fwhm_to_sigma(1.0)
        assert abs(sigma - 1.0 / (2 * math.sqrt(2 * math.log(2)))) < 1e-12


class TestWireHit:
    def test_wire_at_zero(self):
        """Wire centred at 0 should register a hit for small x."""
        x = np.array([0.0, 1e-8, -1e-8])
        hits = wire_hit(x, pitch=1e-3, half_thick=25e-6)
        assert hits.all()

    def test_no_hit_between_wires(self):
        """Midpoint between wires should never be hit."""
        pitch = 1e-3
        x = np.array([pitch / 2 - 1e-9])   # just before next wire
        hits = wire_hit(x, pitch=pitch, half_thick=25e-6)
        assert not hits[0]

    def test_periodicity(self):
        """Hit pattern should repeat every pitch."""
        pitch = 1e-3
        x_ref  = np.array([0.0, 10e-6, -10e-6])
        x_shft = x_ref + 5 * pitch
        assert np.array_equal(
            wire_hit(x_ref,  pitch, 25e-6),
            wire_hit(x_shft, pitch, 25e-6),
        )

    def test_transmission_matches_analytic(self):
        """Sampled hit fraction should match (thick/pitch) within 1 %."""
        pitch     = WP_PITCH
        half_t    = WP_THICK / 2
        x         = np.linspace(0, pitch, 100_000, endpoint=False)
        hit_frac  = wire_hit(x, pitch, half_t).mean()
        expected  = WP_THICK / pitch
        assert abs(hit_frac - expected) < 0.01 * expected


class TestKinematics:
    def test_12C_1MeV_relativistic(self):
        """12C at 1 MeV/u: β ≈ 0.0463, γ ≈ 1.00107."""
        v, beta, gamma = compute_kinematics(A=12, energy_mev_per_u=1.0, relativistic=True)
        assert 0.040 < beta  < 0.055
        assert 1.000 < gamma < 1.005
        assert abs(v - beta * C_LIGHT) < 1.0   # m/s tolerance

    def test_nr_vs_relativistic_low_energy(self):
        """At 1 MeV/u the non-relativistic and relativistic speeds differ < 0.15 %."""
        v_r, _, _ = compute_kinematics(12, 1.0, relativistic=True)
        v_n, _, _ = compute_kinematics(12, 1.0, relativistic=False)
        assert abs(v_r - v_n) / v_r < 0.0015

    def test_high_energy_relativistic_beta_lt_1(self):
        """β must always be < 1."""
        for E in [1, 100, 1000]:
            _, beta, _ = compute_kinematics(1, E, relativistic=True)
            assert beta < 1.0

    def test_analytic_T_wire_plane(self):
        """Wire-plane open-area fraction must equal (pitch - thick) / pitch."""
        T = analytic_T(WP_PITCH, WP_THICK, axes=1)
        assert abs(T - 0.98) < 1e-9

    def test_analytic_T_mesh(self):
        T = analytic_T(GRID1_PITCH, GRID1_THICK, axes=2)
        expected = ((GRID1_PITCH - GRID1_THICK) / GRID1_PITCH) ** 2
        assert abs(T - expected) < 1e-9


# ── Integration / physics checks ─────────────────────────────────────────────

class TestSimulation:
    """End-to-end integration tests."""

    # NOTE: fwhm_xy must be >> wire half-thickness (16 µm for grid1) so the
    # beam spans many wire pitches and the geometric transmission formula
    # applies.  A 3 mm spot (sigma ≈ 1.27 mm >> 1.238 mm pitch) is safe.
    TIGHT_PARAMS = dict(
        N=100_000,
        D_source=0.5,
        fwhm_xy=3e-3,        # 3 mm — realistic spot, spans many wire pitches
        fwhm_angle=1e-4,     # 0.1 mrad — essentially parallel beam
        energy_mev_per_u=1.0,
        A=12, Z=6,
        eta_MCP=1.0,
        eta_IC=1.0,
        n_bins=50,
        seed=42,
        relativistic=True,
        fill_ic_detected=True,
    )

    def run(self, **overrides):
        p = dict(self.TIGHT_PARAMS)
        p.update(overrides)
        h, s, *_ = run_simulation(**p)
        return h, s

    # ── Geometry sanity ──────────────────────────────────────────────────────

    def test_tight_beam_most_pass(self):
        """
        With a realistic 3 mm spot and 0.1 mrad divergence the beam is well
        within all apertures.  Grid1 analytic T ≈ 94.95 %, so more than 80 %
        of generated particles should pass grid1, and a substantial fraction
        should reach IC.
        """
        h, s = self.run()
        frac_g1 = s["N_pass_grid1"] / s["N_generated"]
        assert frac_g1 > 0.80, f"Grid1 pass fraction too low: {frac_g1:.3f}"
        assert s["N_detected_IC"] > 0, "No particle reached IC"

    def test_large_angle_geometric_loss(self):
        """
        Wide divergence should cause heavy geometric losses at apertures.
        With 3 mm spot and 40 mrad FWHM the beam is ~17 mm wide at WP4 —
        comparable to the ±25 mm aperture — so many particles are cut.
        With 0.1 mrad FWHM the beam barely spreads beyond its 3 mm source width.
        """
        _, s_tight = self.run(fwhm_angle=1e-4)   # 0.1 mrad — nearly parallel
        _, s_wide  = self.run(fwhm_angle=0.04)   # 40 mrad FWHM
        assert s_wide["N_detected_IC"] < s_tight["N_detected_IC"], (
            f"Wide beam ({s_wide['N_detected_IC']}) should have fewer IC events "
            f"than tight beam ({s_tight['N_detected_IC']})"
        )

    def test_eta_mcp_efficiency(self):
        """Setting η_MCP = 0.5 should record ~50 % of defined TOF events."""
        _, s = self.run(N=200_000, eta_MCP=0.5, seed=99)
        ratio = s["N_tof_recorded"] / s["N_tof_defined"] if s["N_tof_defined"] else 0
        assert abs(ratio - 0.5) < 0.02, f"η_MCP efficiency wrong: {ratio:.4f}"

    def test_eta_ic_efficiency(self):
        """Setting η_IC = 0.7 should detect ~70 % of geometrically reached events."""
        _, s = self.run(N=200_000, eta_IC=0.7, seed=101)
        ratio = s["N_detected_IC"] / s["N_reach_IC_geometric"] if s["N_reach_IC_geometric"] else 0
        assert abs(ratio - 0.7) < 0.02, f"η_IC efficiency wrong: {ratio:.4f}"

    def test_tof_mean_close_to_analytic(self):
        """
        TOF mean should be close to Δz / v for a nearly parallel beam.
        (With path-length correction the mean is slightly above Δz/v because
         √(1+θ²) ≥ 1;  for σ_angle ≈ 0 the correction is negligible.)
        """
        _, s = self.run(N=100_000, fwhm_angle=1e-6, seed=7)
        v, _, _ = compute_kinematics(12, 1.0, relativistic=True)
        tof_analytic_ns = (Z_GRID2 - Z_GRID1) / v * 1e9
        tof_sim_ns      = s["tof_mean_ns"]
        # Should agree within 0.01 %
        if tof_sim_ns > 0:
            rel_err = abs(tof_sim_ns - tof_analytic_ns) / tof_analytic_ns
            assert rel_err < 1e-4, (
                f"TOF mismatch: sim={tof_sim_ns:.6f} ns, "
                f"analytic={tof_analytic_ns:.6f} ns"
            )

    def test_counts_non_negative(self):
        """All histogram bin counts must be non-negative."""
        h, _ = self.run()
        for key, hist in h.items():
            assert all(c >= 0 for c in hist["counts"]), f"Negative counts in {key}"

    def test_histogram_edges_monotone(self):
        """Bin edges must be strictly increasing."""
        h, _ = self.run()
        for key, hist in h.items():
            edges = hist["edges"]
            assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)), \
                f"Non-monotone edges in {key}"

    def test_n_bins_correct(self):
        """Number of bins in each histogram must equal n_bins."""
        n = 75
        h, _ = self.run(n_bins=n)
        for key, hist in h.items():
            assert len(hist["counts"]) == n, f"Wrong bin count in {key}"
            assert len(hist["edges"])  == n + 1, f"Wrong edge count in {key}"

    def test_reproducible_with_seed(self):
        """Same seed → identical results."""
        h1, s1 = self.run(seed=12345)
        h2, s2 = self.run(seed=12345)
        assert s1["N_detected_IC"] == s2["N_detected_IC"]
        assert h1["tof"]["counts"] == h2["tof"]["counts"]

    def test_different_seeds_differ(self):
        """Different seeds → (almost certainly) different results."""
        # Use N large enough that different seeds almost certainly differ
        _, s1 = self.run(seed=1,  N=50_000)
        _, s2 = self.run(seed=99, N=50_000)
        # With ~50k events and reasonable IC yield, P(equal) is negligible
        assert s1["N_detected_IC"] != s2["N_detected_IC"]

    def test_tof_defined_le_pass_grid1(self):
        """TOF events require a start signal (passing grid1 wire), so N_tof_defined ≤ N_pass_grid1."""
        _, s = self.run()
        assert s["N_tof_defined"] <= s["N_pass_grid1"]

    def test_reach_ic_le_pass_grid2(self):
        """IC count can't exceed grid2 pass count."""
        _, s = self.run()
        assert s["N_reach_IC_geometric"] <= s["N_pass_grid2"]

    def test_tof_recorded_le_tof_defined(self):
        """Recorded TOF events ≤ defined TOF events."""
        _, s = self.run()
        assert s["N_tof_recorded"] <= s["N_tof_defined"]


# ── Wire-plane transmission sanity ────────────────────────────────────────────

class TestAnalyticTransmission:
    def test_wp_is_98_percent(self):
        T = analytic_T(WP_PITCH, WP_THICK, axes=1)
        assert abs(T - 0.98) < 1e-9

    def test_grid1_between_0_and_1(self):
        T = analytic_T(GRID1_PITCH, GRID1_THICK, axes=2)
        assert 0 < T < 1

    def test_grid2_between_0_and_1(self):
        T = analytic_T(GRID2_PITCH, GRID2_THICK, axes=2)
        assert 0 < T < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
