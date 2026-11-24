#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

from __future__ import annotations
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Union, List, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy.fft import next_fast_len, rfft as sp_rfft, irfft as sp_irfft
import h5py
import os
import openmm.unit as unit
from openmm.app import Topology, Element
import warnings
from collections import defaultdict
from itertools import combinations_with_replacement

logger = logging.getLogger(__name__)

# for GPU acceleration
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class Analyzer:
    """
    Analyzer for geometric/topological properties of curves:
    - Writhe of a single curve
    - Writhe between two curves
    - Writhe along a trajectory
    - Radius of gyration (RG)
    """

    @staticmethod
    def _empty_persistent_length_fit() -> dict:
        """Return a consistent payload for fits without valid uncertainty."""
        return {
            'Lp': np.nan,
            'Lp_se': np.nan,
            'Lp_ci95': (np.nan, np.nan),
            'fit_slope': np.nan,
            'fit_slope_se': np.nan,
            'fit_cov': np.array([[np.nan]]),
            'fit_n_points': 0,
            'r_squared': np.nan,
        }

    @staticmethod
    def _fit_persistent_length_from_corr(
        tangent_corr: np.ndarray,
        mean_bond_length: float,
        min_fit_points: int = 10,
        corr_threshold: float = 0.01,
    ) -> dict:
        """
        Fit persistent length from tangent correlation and propagate fit uncertainty.

        The fit uses the existing zero-intercept log-linear model:
        ``log(C(s)) = slope * s`` and ``Lp = -mean_bond_length / slope``.
        The returned covariance is one-dimensional because only ``slope`` is
        fitted.
        """
        tangent_corr = np.asarray(tangent_corr)
        s_vals = np.arange(len(tangent_corr))
        valid = (~np.isnan(tangent_corr)) & (tangent_corr > corr_threshold)
        if len(valid) > 0:
            valid[0] = False

        result = Analyzer._empty_persistent_length_fit()
        n_fit = int(np.sum(valid))
        result['fit_n_points'] = n_fit

        if n_fit < 2:
            return result

        log_C = np.log(tangent_corr[valid])
        s_fit = s_vals[valid]
        x_sq_sum = np.dot(s_fit, s_fit)
        if x_sq_sum <= 0:
            return result

        slope = np.dot(s_fit, log_C) / x_sq_sum
        result['fit_slope'] = slope

        if not np.isfinite(slope):
            return result

        if slope >= 0:
            return result

        Lp = -mean_bond_length / slope
        y_pred = slope * s_fit
        residuals = log_C - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_C - np.mean(log_C))**2)
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 1.0

        dof = n_fit - 1
        slope_var = np.nan
        slope_se = np.nan
        Lp_se = np.nan
        Lp_ci95 = (np.nan, np.nan)
        if dof > 0:
            residual_var = ss_res / dof
            if np.isfinite(residual_var):
                slope_var = residual_var / x_sq_sum
                if slope_var >= 0:
                    slope_se = np.sqrt(slope_var)
                    Lp_se = abs(mean_bond_length / (slope**2)) * slope_se
                    Lp_ci95 = (Lp - 1.96 * Lp_se, Lp + 1.96 * Lp_se)

        result.update(
            {
                'Lp': Lp,
                'Lp_se': Lp_se,
                'Lp_ci95': Lp_ci95,
                'fit_slope_se': slope_se,
                'fit_cov': np.array([[slope_var]]),
                'r_squared': r_squared,
            }
        )
        return result

    @staticmethod
    def _segment_solid_angle(
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> float:
        r13, r14 = q1 - p1, q2 - p1
        r23, r24 = q1 - p2, q2 - p2

        n1 = np.cross(r13, r14)
        n2 = np.cross(r14, r24)
        n3 = np.cross(r24, r23)
        n4 = np.cross(r23, r13)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        n3 /= np.linalg.norm(n3)
        n4 /= np.linalg.norm(n4)

        angles = np.arcsin(
            [
                np.clip(np.dot(n1, n2), -1, 1),
                np.clip(np.dot(n2, n3), -1, 1),
                np.clip(np.dot(n3, n4), -1, 1),
                np.clip(np.dot(n4, n1), -1, 1),
            ]
        )

        omega_star = np.sum(angles)
        sign = np.sign(np.dot(np.cross(q2 - q1, p2 - p1), r13))

        return float((omega_star / (4 * np.pi)) * sign)

    @staticmethod
    def _segment_solid_angle_batch(
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> np.ndarray:
        """Vectorized Gauss linking solid angle for batches of segment pairs."""
        r13 = q1 - p1
        r14 = q2 - p1
        r23 = q1 - p2
        r24 = q2 - p2

        n1 = np.cross(r13, r14)
        n2 = np.cross(r14, r24)
        n3 = np.cross(r24, r23)
        n4 = np.cross(r23, r13)

        norm1 = np.linalg.norm(n1, axis=-1, keepdims=True)
        norm2 = np.linalg.norm(n2, axis=-1, keepdims=True)
        norm3 = np.linalg.norm(n3, axis=-1, keepdims=True)
        norm4 = np.linalg.norm(n4, axis=-1, keepdims=True)

        valid = (
            (norm1[..., 0] > 0)
            & (norm2[..., 0] > 0)
            & (norm3[..., 0] > 0)
            & (norm4[..., 0] > 0)
        )

        n1_u = np.divide(n1, norm1, out=np.zeros_like(n1), where=norm1 > 0)
        n2_u = np.divide(n2, norm2, out=np.zeros_like(n2), where=norm2 > 0)
        n3_u = np.divide(n3, norm3, out=np.zeros_like(n3), where=norm3 > 0)
        n4_u = np.divide(n4, norm4, out=np.zeros_like(n4), where=norm4 > 0)

        d12 = np.clip(np.einsum("...i,...i->...", n1_u, n2_u), -1.0, 1.0)
        d23 = np.clip(np.einsum("...i,...i->...", n2_u, n3_u), -1.0, 1.0)
        d34 = np.clip(np.einsum("...i,...i->...", n3_u, n4_u), -1.0, 1.0)
        d41 = np.clip(np.einsum("...i,...i->...", n4_u, n1_u), -1.0, 1.0)

        omega_star = (
            np.arcsin(d12) + np.arcsin(d23) + np.arcsin(d34) + np.arcsin(d41)
        )
        sign = np.sign(np.einsum("...i,...i->...", np.cross(q2 - q1, p2 - p1), r13))
        result = (omega_star / (4.0 * np.pi)) * sign
        return np.where(valid, result, 0.0)

    @classmethod
    def _compute_writhe_single_curve_legacy(
        cls,
        coords: np.ndarray,
        closed: bool = True,
    ) -> float:
        N = len(coords)
        if closed:
            coords = np.vstack((coords, coords[0]))  # Close the loop
        n_seg = N if closed else max(N - 1, 0)

        writhe = 0.0
        for i in range(n_seg):
            for j in range(i + 1, n_seg):
                if abs(i - j) > 1 and not (closed and {i, j} == {0, n_seg - 1}):
                    writhe += cls._segment_solid_angle(
                        coords[i], coords[i + 1], coords[j], coords[j + 1]
                    )
        return float(writhe)

    @classmethod
    def _compute_writhe_single_curve_vectorized(
        cls,
        coords: np.ndarray,
        closed: bool = True,
        block_size: int = 128,
    ) -> float:
        coords = np.asarray(coords, dtype=np.float64)
        N = len(coords)
        n_seg = N if closed else max(N - 1, 0)
        if n_seg < 3:
            return 0.0

        p1 = coords[:n_seg]
        p2 = np.roll(coords, -1, axis=0)[:n_seg] if closed else coords[1 : n_seg + 1]

        writhe = 0.0
        for i0 in range(0, n_seg, block_size):
            i1 = min(i0 + block_size, n_seg)
            i_idx = np.arange(i0, i1)[:, None]
            for j0 in range(i0, n_seg, block_size):
                j1 = min(j0 + block_size, n_seg)
                j_idx = np.arange(j0, j1)[None, :]
                mask = (j_idx - i_idx) > 1
                if closed:
                    mask &= ~((i_idx == 0) & (j_idx == n_seg - 1))
                ii, jj = np.nonzero(mask)
                if ii.size == 0:
                    continue
                pair_vals = cls._segment_solid_angle_batch(
                    p1[i0 + ii], p2[i0 + ii], p1[j0 + jj], p2[j0 + jj]
                )
                writhe += float(np.sum(pair_vals))
        return float(writhe)

    @classmethod
    def compute_writhe_single_curve(
        cls,
        coords: np.ndarray,
        closed: bool = True,
        method: str = "vectorized",
    ) -> float:
        if method == "vectorized":
            return cls._compute_writhe_single_curve_vectorized(coords, closed=closed)
        if method == "legacy":
            return cls._compute_writhe_single_curve_legacy(coords, closed=closed)
        raise ValueError("method must be 'vectorized' or 'legacy'")

    @classmethod
    def _compute_writhe_between_curves_legacy(
        cls,
        curve1: np.ndarray,
        curve2: np.ndarray,
    ) -> float:
        N, M = len(curve1), len(curve2)
        curve1_closed = np.vstack([curve1, curve1[0]])
        curve2_closed = np.vstack([curve2, curve2[0]])

        writhe = 0.0
        for i in range(N):
            for j in range(M):
                writhe += cls._segment_solid_angle(
                    curve1_closed[i],
                    curve1_closed[i + 1],
                    curve2_closed[j],
                    curve2_closed[j + 1],
                )

        return float(writhe)

    @classmethod
    def _compute_writhe_between_curves_vectorized(
        cls,
        curve1: np.ndarray,
        curve2: np.ndarray,
        block_size: int = 128,
    ) -> float:
        curve1 = np.asarray(curve1, dtype=np.float64)
        curve2 = np.asarray(curve2, dtype=np.float64)
        N, M = len(curve1), len(curve2)
        if N == 0 or M == 0:
            return 0.0

        p1 = curve1
        p2 = np.roll(curve1, -1, axis=0)
        q1 = curve2
        q2 = np.roll(curve2, -1, axis=0)

        writhe = 0.0
        for i0 in range(0, N, block_size):
            i1 = min(i0 + block_size, N)
            for j0 in range(0, M, block_size):
                j1 = min(j0 + block_size, M)
                pair_vals = cls._segment_solid_angle_batch(
                    p1[i0:i1][:, None, :],
                    p2[i0:i1][:, None, :],
                    q1[j0:j1][None, :, :],
                    q2[j0:j1][None, :, :],
                )
                writhe += float(np.sum(pair_vals))
        return float(writhe)

    @classmethod
    def compute_writhe_between_curves(
        cls,
        curve1: np.ndarray,
        curve2: np.ndarray,
        method: str = "vectorized",
    ) -> float:
        """
        Computes the writhe between two closed curves (curve1 and curve2).

        Parameters
        ----------
        curve1, curve2 : ndarray of shape (N, 3) and (M, 3)
            Points defining the closed curves C1 and C2.
        method : {'vectorized', 'legacy'}, default 'vectorized'
            Computation implementation to use.

        Returns
        -------
        float
            The computed writhe between the two curves.
        """
        if method == "vectorized":
            return cls._compute_writhe_between_curves_vectorized(curve1, curve2)
        if method == "legacy":
            return cls._compute_writhe_between_curves_legacy(curve1, curve2)
        raise ValueError("method must be 'vectorized' or 'legacy'")

    @classmethod
    def compute_writhe_trajectory(
        cls,
        trajectory: np.ndarray,
        closed: bool = True,
        processes: Optional[int] = None,
        method: str = "vectorized",
    ) -> List[float]:
        if method not in {"vectorized", "legacy"}:
            raise ValueError("method must be 'vectorized' or 'legacy'")
        if processes is None:
            processes = max(cpu_count() - 1, 1)

        with Pool(processes) as pool:
            args = [(frame, closed, method) for frame in trajectory]
            results = pool.starmap(cls.compute_writhe_single_curve, args)

        return results

    @staticmethod
    def compute_RG(
        positions: np.ndarray, 
        return_components: bool = False,
        bead_masses: Optional[np.ndarray] = None,
        print_choice: bool = False,
    ) -> Union[float, np.ndarray, Tuple]:
        """
        Calculates the Radius of Gyration (Rg).

        Args:
            positions: Coordinates array of shape (N, 3) or (T, N, 3).
            return_components: If True, returns a tuple (rg_total, rg_xyz).
                               rg_xyz will contain [rg_x, rg_y, rg_z].
            bead_masses: Optional array of bead masses for mass-weighted Rg.
        """
        # receive traj.xyz() as positions
        positions = np.asarray(positions)

        # 1. Handle masses and log choice
        if bead_masses is not None:
            masses = np.asarray(bead_masses)
            if print_choice:
                logger.info("Computing mass-weighted Radius of Gyration")
        else:
            masses = np.ones(positions.shape[-2])
            if print_choice:
                logger.info("Computing geometric Radius of Gyration (uniform mass)")

        # ---------------------------------------------------------
        # Case 1: Single Frame (N, 3)
        # ---------------------------------------------------------
        if positions.ndim == 2:
            # 1. Calculate Center of Mass
            center_of_mass = np.average(positions, axis=0, weights=masses)

            # 2. Calculate squared deviations for each dimension (x, y, z) separately
            # Shape remains (N, 3) here
            sq_deviations = (positions - center_of_mass) ** 2

            # 3. Average over particles (N) to get squared Rg components
            # Shape becomes (3,) -> [Rgx^2, Rgy^2, Rgz^2]
            rg_sq_components = np.average(sq_deviations, axis=0, weights=masses)

            # 4. Calculate Total Rg
            # Rg = sqrt(Rgx^2 + Rgy^2 + Rgz^2)
            rg_total = float(np.sqrt(np.sum(rg_sq_components)))

            if return_components:
                rg_xyz = np.sqrt(rg_sq_components)  # Shape (3,)
                return rg_total, rg_xyz
            else:
                return rg_total

        # ---------------------------------------------------------
        # Case 2: Trajectory (T, N, 3)
        # ---------------------------------------------------------
        elif positions.ndim == 3:
            # 1. Calculate Centers of Mass
            # Shape (T, 3)
            centers_of_mass = np.average(positions, axis=1, weights=masses)

            # 2. Calculate squared deviations
            # Use broadcasting: (T, N, 3) - (T, 1, 3)
            sq_deviations = (positions - centers_of_mass[:, None, :]) ** 2

            # 3. Average over particles (axis 1) to get squared Rg components
            # Shape becomes (T, 3)
            rg_sq_components = np.average(sq_deviations, axis=1, weights=masses)

            # 4. Calculate Total Rg per frame
            # Sum over xyz (axis 1 of the component array), then sqrt
            # Shape (T,)
            rg_total = np.sqrt(np.sum(rg_sq_components, axis=1))

            if return_components:
                rg_xyz = np.sqrt(rg_sq_components)  # Shape (T, 3)
                return rg_total, rg_xyz
            else:
                return rg_total

        else:
            raise ValueError(
                f"positions must have shape (N, 3) or (T, N, 3), got {positions.shape}"
            )

    @staticmethod
    def compute_tangent_correlation(
        positions: np.ndarray,
        max_s: Optional[int] = None,
        bond_indices: Optional[np.ndarray] = None,
        device: str = 'gpu',
        box_vectors: Optional[np.ndarray] = None,
    ) -> dict:
        R"""
        Compute the tangent-tangent correlation function and estimate persistent length.

        The tangent correlation function is defined as:
            C(s) = <û_i · û_{i+s}>
        where û_i is the unit tangent (bond) vector between bead i and i+1,
        and s is the bond index separation.

        For a Worm-Like Chain (WLC) model, C(s) = exp(-s·b / Lp),
        where b is the mean bond length and Lp is the persistent length.

        Parameters
        ----------
        positions : np.ndarray
            Coordinates array of shape (N, 3) for a single frame,
            or (T, N, 3) for a trajectory.
        max_s : int, optional
            Maximum bond separation to compute.  If ``None``, determined
            automatically from two competing constraints:

            - **Upper bound** (noise): ``int(n_bonds * NOISE_FRAC)`` where
              ``NOISE_FRAC = 0.5``.  At larger separations, too few tangent
              pairs contribute per C(s) value → noisy.
            - **Lower bound** (fit quality): ``MIN_FIT_POINTS = 10``.
              Fewer data points make the exponential fit unreliable.

            If upper < lower, the segment is too short for a robust Lp
            estimate; C(s) is still computed up to the upper bound but
            ``fit_reliable`` is set to ``False``.

            A hard cap of 200 is always applied for computational cost.
        bond_indices : np.ndarray, optional
            If provided, only these bond indices (into the full bond array)
            are used. This allows computing tangent correlations for a
            *subset* of bonds (e.g., within a contiguous type segment).
            Bond index i corresponds to the vector r_{i+1} - r_i.
        device : str, default ``'gpu'``
            Compute backend.  ``'gpu'`` uses CuPy for accelerated
            bond-vector / dot-product computation (requires CuPy).
            The final Lp fit is always performed on CPU.
        box_vectors : np.ndarray, optional
            Periodic box vectors to apply Minimum Image Convention (MIC)
            for bond vectors. Shape ``(3,)``, ``(3, 3)``, or ``(T, 3, 3)``.

        Returns
        -------
        dict
            'tangent_corr'    : np.ndarray of shape (max_s + 1,)
                                C(s) for s = 0, 1, ..., max_s.  C(0) = 1.
            'contour_length'  : np.ndarray of shape (max_s + 1,)
                                Cumulative contour distance = s * mean_bond_length.
            'Lp'              : float or np.nan
                                Fitted persistent length (units of position).
            'Lp_se'           : float or np.nan
                                Fit-based standard error of ``Lp`` from
                                covariance propagation.
            'Lp_ci95'         : tuple of float
                                Approximate 95% confidence interval for ``Lp``.
            'mean_bond_length': float
                                Average bond length over all selected bonds/frames.
            'fit_reliable'    : bool
                                ``True`` if max_s >= MIN_FIT_POINTS, i.e. enough
                                data points were available for a robust fit.
            'r_squared'       : float or np.nan
                                Coefficient of determination (R^2) for the fit.
            'fit_slope'       : float or np.nan
                                Fitted zero-intercept slope of ``log(C(s))`` vs ``s``.
            'fit_slope_se'    : float or np.nan
                                Standard error of ``fit_slope``.
            'fit_cov'         : np.ndarray
                                One-by-one covariance matrix for ``fit_slope``.
            'fit_n_points'    : int
                                Number of positive correlation points used for fitting.
            'fit_details'     : str
                                Detailed reason for the reliability status.
        """
        positions = np.asarray(positions)

        # ---- Unify to (T, N, 3) ----
        if positions.ndim == 2:
            positions = positions[np.newaxis, :, :]  # (1, N, 3)
        elif positions.ndim != 3:
            raise ValueError(
                f"positions must have shape (N, 3) or (T, N, 3), got {positions.shape}"
            )

        # ---- Select compute backend ----
        use_gpu = (device.lower() == 'gpu')
        if use_gpu and not CUPY_AVAILABLE:
            logger.warning("CuPy not available. Falling back to CPU.")
            use_gpu = False
        xp = cp if use_gpu else np

        # Transfer to device
        positions_d = xp.asarray(positions)

        # ---- Bond vectors & unit tangents ----
        # bond_vectors[:, i, :] = positions[:, i+1, :] - positions[:, i, :]
        all_bond_vectors = positions_d[:, 1:, :] - positions_d[:, :-1, :]  # (T, N-1, 3)

        # ---- Apply Minimum Image Convention (MIC) if box provided ----
        if box_vectors is not None:
            box_d = xp.asarray(box_vectors)
            if box_d.ndim == 3:
                # (T, 3, 3) -> diagonal -> (T, 1, 3)
                box_diag = xp.diagonal(box_d, axis1=1, axis2=2)[:, xp.newaxis, :]
            elif box_d.ndim == 1:
                # (3,) -> (1, 1, 3)
                box_diag = box_d[xp.newaxis, xp.newaxis, :]
            elif box_d.ndim == 2 and box_d.shape == (3, 3):
                # (3, 3) -> (1, 1, 3)
                box_diag = xp.diagonal(box_d)[:, xp.newaxis, :]
            else:
                raise ValueError("box_vectors must be (3,), (3, 3), or (T, 3, 3)")
            
            # MIC: delta = delta - box_diag * round(delta / box_diag)
            all_bond_vectors -= box_diag * xp.round(all_bond_vectors / box_diag)

        all_bond_lengths = xp.linalg.norm(
            all_bond_vectors, axis=-1, keepdims=True
        )  # (T, N-1, 1)

        # Avoid division by zero for degenerate bonds
        safe_lengths = xp.where(all_bond_lengths > 0, all_bond_lengths, 1.0)
        all_unit_tangents = all_bond_vectors / safe_lengths  # (T, N-1, 3)

        # ---- Subset selection ----
        if bond_indices is not None:
            bond_indices_np = np.asarray(bond_indices)
            # For segment-based computation, bond_indices must be *contiguous*
            # so that index separations still correspond to chain separations.
            if use_gpu:
                bi = cp.asarray(bond_indices_np)
            else:
                bi = bond_indices_np
            unit_tangents = all_unit_tangents[:, bi, :]       # (T, M, 3)
            bond_lengths_sel = all_bond_lengths[:, bi, 0]     # (T, M)
        else:
            unit_tangents = all_unit_tangents
            bond_lengths_sel = all_bond_lengths[:, :, 0]

        n_bonds = unit_tangents.shape[1]  # M

        # ---- Determine max_s: dual-bound logic ----
        # Upper bound (noise): don't exceed this fraction of n_bonds so that
        #   each C(s) is averaged over enough tangent pairs.
        # Lower bound (fit):   need at least this many s-values for a
        #   reliable exponential fit of ln C(s).
        NOISE_FRAC = 0.5
        MIN_FIT_POINTS = 10
        GLOBAL_CAP = 200  # absolute cap for computational cost

        max_s_from_noise = max(1, int(n_bonds * NOISE_FRAC))

        if max_s is None:
            max_s = min(max_s_from_noise, GLOBAL_CAP)
        else:
            # User-specified: only apply the hard physical limit
            max_s = min(max_s, n_bonds - 1)

        # Can we meet the fit-quality requirement?
        fit_reliable = (max_s >= MIN_FIT_POINTS)

        if max_s < 1:
            # Not enough bonds to compute any correlation
            mean_bl = float(xp.mean(bond_lengths_sel))
            if use_gpu:
                del positions_d, all_bond_vectors, all_bond_lengths
                del safe_lengths, all_unit_tangents, unit_tangents, bond_lengths_sel
                cp.get_default_memory_pool().free_all_blocks()
            return {
                'tangent_corr': np.array([1.0]),
                'contour_length': np.array([0.0]),
                'mean_bond_length': mean_bl,
                'fit_reliable': False,
                'fit_details': "Segment too short to compute correlation.",
                **Analyzer._empty_persistent_length_fit(),
            }

        mean_bond_length = float(xp.mean(bond_lengths_sel))

        # ---- Tangent correlation C(s) ----
        tangent_corr_d = xp.zeros(max_s + 1)
        tangent_corr_d[0] = 1.0  # C(0) = <û · û> = 1

        for s in range(1, max_s + 1):
            # Dot products for all valid pairs and all frames
            dots = xp.sum(
                unit_tangents[:, :n_bonds - s, :] * unit_tangents[:, s:, :],
                axis=-1,
            )  # (T, n_bonds - s)
            tangent_corr_d[s] = xp.mean(dots)

        # ---- Transfer result to CPU for fitting ----
        if use_gpu:
            tangent_corr = cp.asnumpy(tangent_corr_d)
            # Free GPU memory
            del positions_d, all_bond_vectors, all_bond_lengths
            del safe_lengths, all_unit_tangents, unit_tangents
            del bond_lengths_sel, tangent_corr_d
            cp.get_default_memory_pool().free_all_blocks()
        else:
            tangent_corr = np.asarray(tangent_corr_d)

        # ---- Fit persistent length via log-linear regression (CPU) ----
        # C(s) = exp(-s * b / Lp)  =>  ln C(s) = -(b / Lp) * s
        s_vals = np.arange(max_s + 1)
        contour_length = s_vals * mean_bond_length

        fit_details = ""
        if not fit_reliable:
            fit_details = f"Segment too short (max_s={max_s} < {MIN_FIT_POINTS}). "

        if max_s >= 1 and tangent_corr[1] <= 0:
            fit_reliable = False
            fit_details += (
                "C(1) <= 0: Chain exhibits local anti-correlation or strong "
                "flexibility. WLC model is inapplicable."
            )

        fit_result = Analyzer._fit_persistent_length_from_corr(
            tangent_corr,
            mean_bond_length=mean_bond_length,
            min_fit_points=MIN_FIT_POINTS,
        )

        if fit_result['fit_n_points'] < 2:
            if fit_reliable:
                fit_reliable = False
                fit_details += "Not enough positive C(s) data points for fitting."
        elif not np.isfinite(fit_result['Lp']):
            if fit_reliable:
                fit_reliable = False
                fit_details += "Persistent length fit produced a non-finite result."
        elif fit_result['fit_slope'] >= 0:
            if fit_reliable:
                fit_reliable = False
                fit_details += "Non-decaying correlation (slope >= 0)."
        elif fit_reliable and fit_result['r_squared'] < 0.8:
            fit_reliable = False
            fit_details += f"Poor exponential fit (R^2 = {fit_result['r_squared']:.2f})."

        if fit_reliable and not fit_details:
            fit_details = "Fit is reliable."

        return {
            'tangent_corr': tangent_corr,
            'contour_length': contour_length,
            'mean_bond_length': mean_bond_length,
            'fit_reliable': fit_reliable,
            'fit_details': fit_details.strip(),
            **fit_result,
        }


    @staticmethod
    def wrap_coordinates(positions: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
        """
        Convert Unwrapped coordinates to Wrapped coordinates (inside the box).

        Args:
            positions: (..., 3) array
            box_vectors: (3, 3) array or (3,) array of box lengths

        Returns:
            positions_wrapped: Coordinates within [0, box_length]
        """
        # Handle cubic box usually stored as [Lx, Ly, Lz] or diagonals of 3x3
        if box_vectors.shape == (3, 3):
            box_diag = np.diag(box_vectors)
        else:
            box_diag = np.array(box_vectors)

        # Modulo operation handles the wrapping
        # positions % box_diag ensures result is in [0, L)
        return positions % box_diag

    # =========================================================================
    # Public Interface: VACF
    # =========================================================================
    @staticmethod
    def calculate_vacf(
        velocities: np.ndarray,
        bead_types: np.ndarray,
        sampling_step: int = 1,
        platform: str = "auto",
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Velocity Autocorrelation Function (VACF).

        Parameters
        ----------
        velocities : np.ndarray
            Shape (n_frames, n_beads, 3)
        bead_types : np.ndarray
            Shape (n_beads,)
        sampling_step : int
            Step size for sampling frames.
        platform : str
            'auto', 'CPU', or 'CUDA'.
            'auto' will use GPU if available, else CPU.

        Returns
        -------
        Dict[str, np.ndarray]
            VACF curves for 'general' and each bead type.
        """
        use_gpu = Analyzer._check_platform(platform)
        return Analyzer._calculate_vacf_impl(
            velocities, bead_types, sampling_step, use_gpu=use_gpu
        )

    # =========================================================================
    # Public Interface: Spatial Velocity Correlation
    # =========================================================================
    @staticmethod
    def calculate_spatial_vel_corr(
        coords: np.ndarray,
        velocities: np.ndarray,
        bead_types: np.ndarray,
        dist_range: float,
        sampling_step: int = 1,
        num_bins: int = 100,
        platform: str = "auto",
        chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate spatial velocity correlation C(r) = <v_i . v_j>.

        Parameters
        ----------
        coords : np.ndarray
            Shape (n_frames, n_beads, 3)
        velocities : np.ndarray
            Shape (n_frames, n_beads, 3)
        bead_types : np.ndarray
            Shape (n_beads,)
        dist_range : float
            Maximum distance for correlation.
        sampling_step : int
            Step size for sampling frames.
        num_bins : int
            Number of bins for distance.
        platform : str
            'auto', 'CPU', or 'CUDA'.
        chunk_size : int, optional
            Number of beads to process simultaneously to cap memory usage.

        Returns
        -------
        Dict[str, np.ndarray]
            Correlation curves and 'bin_centers'.
        """
        use_gpu = Analyzer._check_platform(platform)
        return Analyzer._calculate_spatial_vel_corr_impl(
            coords,
            velocities,
            bead_types,
            dist_range,
            sampling_step=sampling_step,
            num_bins=num_bins,
            chunk_size=chunk_size,
            use_gpu=use_gpu,
        )

    # =========================================================================
    # Helper: Platform Check
    # =========================================================================
    @staticmethod
    def _check_platform(platform: str) -> bool:
        """Returns True if GPU should be used, False otherwise."""
        if platform.upper() == "CUDA" or platform.upper() == "GPU":
            if not CUPY_AVAILABLE:
                warnings.warn(
                    "CUDA requested but CuPy not installed. Falling back to CPU."
                )
                return False
            return True
        elif platform.upper() == "CPU":
            return False
        else:  # 'auto'
            if CUPY_AVAILABLE:
                try:
                    if cp.cuda.runtime.getDeviceCount() > 0:
                        return True
                except Exception:
                    pass
            return False

    # =========================================================================
    # Unified Backend Implementation: VACF (CPU / GPU)
    # =========================================================================

    @staticmethod
    def _autocorr_fft(x_multi_dim, use_gpu: bool = False):
        """FFT-based autocorrelation helper shared by CPU and GPU backends."""
        N = x_multi_dim.shape[0]
        if N == 0:
            return x_multi_dim.copy()
        xp = cp if use_gpu else np
        is_complex = bool(xp.iscomplexobj(x_multi_dim))

        if use_gpu:
            n_fft = next_fast_len(2 * N)
            if is_complex:
                F = cp.fft.fft(x_multi_dim, n=n_fft, axis=0)
                res = cp.fft.ifft(F * F.conjugate(), axis=0)[:N, ...].real
            else:
                F = cp.fft.rfft(x_multi_dim, n=n_fft, axis=0)
                res = cp.fft.irfft(F.real**2 + F.imag**2, n=n_fft, axis=0)[:N, ...]
        else:
            n_fft = next_fast_len(2 * N, real=not is_complex)
            if is_complex:
                F = np.fft.fft(x_multi_dim, n=n_fft, axis=0)
                res = np.fft.ifft(F * F.conjugate(), axis=0)[:N, ...].real
            else:
                F = sp_rfft(x_multi_dim, n=n_fft, axis=0, workers=-1)
                res = sp_irfft(
                    F.real**2 + F.imag**2, n=n_fft, axis=0, workers=-1
                )[:N, ...]

        norm_shape = [N] + [1] * (x_multi_dim.ndim - 1)
        norm = (N - xp.arange(0, N)).reshape(norm_shape)
        return res / norm

    @staticmethod
    def _autocorrFFT_gpu(x_multi_dim: "cp.ndarray") -> "cp.ndarray":
        """(GPU) FFT-based autocorrelation helper."""
        return Analyzer._autocorr_fft(x_multi_dim, use_gpu=True)

    @staticmethod
    def _autocorrFFT_cpu(x_multi_dim: np.ndarray) -> np.ndarray:
        """(CPU) FFT-based autocorrelation helper."""
        return Analyzer._autocorr_fft(x_multi_dim, use_gpu=False)

    @staticmethod
    def _calculate_vacf_impl(
        velocities: np.ndarray,
        bead_types: np.ndarray,
        sampling_step: int,
        use_gpu: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Shared CPU/GPU implementation of VACF."""
        logger.info("Calculating VACF on %s...", "GPU" if use_gpu else "CPU")
        xp = cp if use_gpu else np
        bead_types_arr = np.asarray(bead_types)
        type_indices = {
            utype: np.where(bead_types_arr == utype)[0]
            for utype in np.unique(bead_types_arr)
        }

        sampled_vels = xp.asarray(velocities[::sampling_step])
        N = sampled_vels.shape[0]
        if N == 0:
            vacf_all_beads = xp.empty((sampled_vels.shape[1], 0), dtype=xp.float64)
        elif xp.iscomplexobj(sampled_vels):
            vacf_components = Analyzer._autocorr_fft(sampled_vels, use_gpu=use_gpu)
            vacf_all_beads = xp.sum(vacf_components, axis=2).T
        else:
            if use_gpu:
                n_fft = next_fast_len(2 * N)
                F = cp.fft.rfft(sampled_vels, n=n_fft, axis=0)
                power_sum = cp.sum(F.real**2 + F.imag**2, axis=2)
                vacf_sum = cp.fft.irfft(power_sum, n=n_fft, axis=0)[:N, :]
            else:
                n_fft = next_fast_len(2 * N, real=True)
                F = sp_rfft(sampled_vels, n=n_fft, axis=0, workers=-1)
                power_sum = np.sum(F.real**2 + F.imag**2, axis=2)
                vacf_sum = sp_irfft(power_sum, n=n_fft, axis=0, workers=-1)[:N, :]
            norm = (N - xp.arange(0, N))[:, None]
            vacf_all_beads = (vacf_sum / norm).T

        results_xp = {"general": xp.mean(vacf_all_beads, axis=0)}
        for btype, indices in type_indices.items():
            if len(indices) > 0:
                idx_xp = xp.asarray(indices) if use_gpu else indices
                results_xp[btype] = xp.mean(vacf_all_beads[idx_xp, :], axis=0)

        if use_gpu:
            results = {k: cp.asnumpy(v) for k, v in results_xp.items()}
            del sampled_vels, vacf_all_beads, results_xp
            if hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()
            return results
        return results_xp

    @staticmethod
    def _calculate_vacf_gpu(velocities, bead_types, sampling_step):
        """(GPU) Implementation of VACF."""
        return Analyzer._calculate_vacf_impl(
            velocities, bead_types, sampling_step, use_gpu=True
        )

    @staticmethod
    def _calculate_vacf_cpu(velocities, bead_types, sampling_step):
        """(CPU) Implementation of VACF."""
        return Analyzer._calculate_vacf_impl(
            velocities, bead_types, sampling_step, use_gpu=False
        )

    # =========================================================================
    # Unified Backend Implementation: Spatial Velocity Correlation (CPU / GPU)
    # =========================================================================

    @staticmethod
    def _calculate_spatial_vel_corr_impl(
        coords: np.ndarray,
        velocities: np.ndarray,
        bead_types: np.ndarray,
        dist_range: float,
        sampling_step: int = 1,
        num_bins: int = 100,
        chunk_size: Optional[int] = None,
        use_gpu: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Vectorized CPU/GPU spatial velocity correlation without inner Python pair loops."""
        logger.info(
            "Calculating Spatial Velocity Correlation on %s...",
            "GPU" if use_gpu else "CPU",
        )
        xp = cp if use_gpu else np

        n_frames, n_beads, _ = coords.shape
        bins = np.linspace(0, dist_range, num_bins + 1, dtype=np.float32)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        bead_types_arr = np.asarray(bead_types)
        if bead_types_arr.shape[0] != n_beads:
            raise ValueError(
                f"bead_types length ({bead_types_arr.shape[0]}) must match number of beads ({n_beads})."
            )

        unique_types, type_ids = np.unique(bead_types_arr, return_inverse=True)
        type_ids = type_ids.ravel().astype(np.int32)
        type_strs = [str(t) for t in unique_types]
        n_types = len(type_strs)
        type_pairs = sorted(
            {
                "-".join(sorted((type_strs[a], type_strs[b])))
                for a in range(n_types)
                for b in range(a, n_types)
            }
        )
        n_pairs = len(type_pairs)
        pair_to_id = {key: idx for idx, key in enumerate(type_pairs)}
        pair_lut = np.empty((n_types, n_types), dtype=np.int32)
        for a in range(n_types):
            for b in range(n_types):
                pair_lut[a, b] = pair_to_id["-".join(sorted((type_strs[a], type_strs[b])))]

        if chunk_size is None:
            chunk_size = n_beads
        elif not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")

        bins_xp = xp.asarray(bins)
        bin_edges_right = bins_xp[1:]
        type_ids_xp = xp.asarray(type_ids)
        pair_lut_xp = xp.asarray(pair_lut)
        pair_corr = xp.zeros((n_pairs, num_bins), dtype=xp.float64)
        pair_counts = xp.zeros((n_pairs, num_bins), dtype=xp.int64)
        total_buckets = n_pairs * num_bins

        if n_beads >= 2 and n_pairs > 0 and dist_range > 0:
            for frame_idx in range(0, n_frames, sampling_step):
                if use_gpu:
                    frame_coords = cp.asarray(coords[frame_idx], dtype=cp.float32)
                    frame_vels = cp.asarray(velocities[frame_idx], dtype=cp.float32)
                else:
                    frame_coords = np.asarray(coords[frame_idx])
                    frame_vels = np.asarray(velocities[frame_idx])

                for start_i in range(0, n_beads, chunk_size):
                    end_i = min(start_i + chunk_size, n_beads)
                    coords_i = frame_coords[start_i:end_i]
                    vels_i = frame_vels[start_i:end_i]
                    row_global_base = xp.arange(start_i, end_i)[:, None]

                    for start_j in range(start_i, n_beads, chunk_size):
                        end_j = min(start_j + chunk_size, n_beads)
                        coords_j = frame_coords[start_j:end_j]
                        vels_j = frame_vels[start_j:end_j]

                        dists = xp.linalg.norm(
                            coords_i[:, None, :] - coords_j[None, :, :], axis=2
                        )
                        dots = vels_i @ vels_j.T
                        valid = dists < bins_xp[-1]
                        if start_j == start_i:
                            col_global_base = xp.arange(start_j, end_j)[None, :]
                            valid &= row_global_base < col_global_base

                        rows, cols = xp.nonzero(valid)
                        if rows.size == 0:
                            continue

                        d_val = dists[rows, cols]
                        v_val = dots[rows, cols]
                        b_idx = xp.searchsorted(bin_edges_right, d_val, side="right")
                        in_range = b_idx < num_bins
                        rows = rows[in_range]
                        cols = cols[in_range]
                        b_idx = b_idx[in_range]
                        v_val = v_val[in_range]
                        if rows.size == 0:
                            continue

                        p_idx = pair_lut_xp[
                            type_ids_xp[start_i + rows], type_ids_xp[start_j + cols]
                        ]
                        flat_idx = p_idx * num_bins + b_idx
                        pair_corr += xp.bincount(
                            flat_idx,
                            weights=v_val.astype(xp.float64),
                            minlength=total_buckets,
                        ).reshape(n_pairs, num_bins)
                        pair_counts += xp.bincount(
                            flat_idx, minlength=total_buckets
                        ).reshape(n_pairs, num_bins)

        if use_gpu:
            pair_corr_np = cp.asnumpy(pair_corr)
            pair_counts_np = cp.asnumpy(pair_counts)
            del bins_xp, type_ids_xp, pair_lut_xp, pair_corr, pair_counts
            if hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()
        else:
            pair_corr_np = pair_corr
            pair_counts_np = pair_counts

        general_corr = np.sum(pair_corr_np, axis=0)
        general_counts = np.sum(pair_counts_np, axis=0)

        results: Dict[str, np.ndarray] = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            gen = general_corr / general_counts
        gen[general_counts == 0] = np.nan
        results["general"] = gen

        for idx, key in enumerate(type_pairs):
            cnt = pair_counts_np[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = pair_corr_np[idx] / cnt
            corr[cnt == 0] = np.nan
            results[key] = corr

        results["bin_centers"] = bin_centers
        return results

    @staticmethod
    def _calculate_spatial_vel_corr_gpu(
        coords: np.ndarray,
        velocities: np.ndarray,
        bead_types: np.ndarray,
        dist_range: float,
        sampling_step: int = 1,
        num_bins: int = 50,
        chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """(GPU) Spatial velocity correlation wrapper."""
        return Analyzer._calculate_spatial_vel_corr_impl(
            coords,
            velocities,
            bead_types,
            dist_range,
            sampling_step=sampling_step,
            num_bins=num_bins,
            chunk_size=chunk_size,
            use_gpu=True,
        )

    @staticmethod
    def _calculate_spatial_vel_corr_cpu(
        coords, velocities, bead_types, dist_range, sampling_step, num_bins, chunk_size=None
    ):
        """(CPU) Spatial velocity correlation wrapper."""
        return Analyzer._calculate_spatial_vel_corr_impl(
            coords,
            velocities,
            bead_types,
            dist_range,
            sampling_step=sampling_step,
            num_bins=num_bins,
            chunk_size=chunk_size,
            use_gpu=False,
        )

    @staticmethod
    def _msd_fft_batch(
        coords: np.ndarray,
        batch_size: int,
        use_gpu: bool = False,
        free_gpu_pool: bool = True,
    ) -> np.ndarray:
        """
        Shared CPU/GPU FFT-based MSD computation.

        Subtracts each bead's initial position in float64 before FFT/prefix-sum
        evaluation and accumulates S1 in float64 to avoid catastrophic cancellation
        on trajectories with large coordinate offsets or long runs.
        """
        M, B_total, D = coords.shape
        out_dtype = np.result_type(coords.dtype, np.float32)
        if M == 0 or B_total == 0:
            return np.empty((M, B_total), dtype=out_dtype)

        msd_result = np.zeros((M, B_total), dtype=out_dtype)
        if M == 1:
            return msd_result

        xp = cp if use_gpu else np
        n_fft = next_fast_len(2 * M) if use_gpu else next_fast_len(2 * M, real=True)
        den = (M - np.arange(M, dtype=np.float64))[:, None]
        den_xp = xp.asarray(den) if use_gpu else den
        zero_pad = xp.zeros((1, 1), dtype=xp.float64)

        try:
            for start_idx in range(0, B_total, batch_size):
                end_idx = min(start_idx + batch_size, B_total)
                r_batch_f64 = np.asarray(
                    coords[:, start_idx:end_idx, :], dtype=np.float64
                )
                # Shift by frame 0 in float64 so large global offsets do not cancel displacements.
                r_shifted_np = r_batch_f64 - r_batch_f64[0:1, :, :]
                B_current = r_shifted_np.shape[1]

                if use_gpu:
                    r_shifted_gpu = cp.asarray(r_shifted_np, dtype=cp.float32)
                    power_sum = cp.zeros(
                        (n_fft // 2 + 1, B_current), dtype=cp.float32
                    )
                    for dim in range(D):
                        F = cp.fft.rfft(r_shifted_gpu[:, :, dim], n=n_fft, axis=0)
                        power_sum += F.real**2 + F.imag**2
                    S2 = (
                        cp.fft.irfft(power_sum, n=n_fft, axis=0)[:M, :].astype(
                            cp.float64
                        )
                        / den_xp
                    )
                    D_sq = cp.sum(r_shifted_gpu.astype(cp.float64) ** 2, axis=2)
                else:
                    F = sp_rfft(r_shifted_np, n=n_fft, axis=0, workers=-1)
                    power_sum = np.sum(F.real**2 + F.imag**2, axis=2)
                    S2 = sp_irfft(power_sum, n=n_fft, axis=0, workers=-1)[:M, :] / den_xp
                    D_sq = np.sum(r_shifted_np**2, axis=2)

                Q_0 = 2.0 * xp.sum(D_sq, axis=0)
                sub_terms = D_sq[:-1, :] + D_sq[M - 1 : 0 : -1, :]
                cum_sub = xp.cumsum(sub_terms, axis=0)
                pad = xp.broadcast_to(zero_pad, (1, B_current))
                Q_m = Q_0 - xp.concatenate((pad, cum_sub), axis=0)
                S1 = Q_m / den_xp

                msd_batch = S1 - 2.0 * S2
                msd_batch[0, :] = 0.0
                if use_gpu:
                    msd_result[:, start_idx:end_idx] = cp.asnumpy(msd_batch).astype(
                        out_dtype, copy=False
                    )
                    del (
                        r_shifted_gpu,
                        power_sum,
                        F,
                        S2,
                        D_sq,
                        sub_terms,
                        cum_sub,
                        Q_m,
                        S1,
                        msd_batch,
                    )
                else:
                    msd_result[:, start_idx:end_idx] = msd_batch.astype(
                        out_dtype, copy=False
                    )
        finally:
            if use_gpu:
                del den_xp, zero_pad
                if free_gpu_pool and hasattr(cp, "get_default_memory_pool"):
                    cp.get_default_memory_pool().free_all_blocks()

        return msd_result

    @staticmethod
    def _msd_fft_cpu_batch(coords: np.ndarray, batch_size: int) -> np.ndarray:
        """(Internal) CPU implementation of MSD using SciPy rFFT."""
        return Analyzer._msd_fft_batch(coords, batch_size, use_gpu=False)

    @staticmethod
    def _msd_fft_gpu_batch(coords: np.ndarray, batch_size: int) -> np.ndarray:
        """(Internal) GPU implementation of MSD using CuPy rFFT."""
        return Analyzer._msd_fft_batch(coords, batch_size, use_gpu=True)

    @staticmethod
    def compute_msd(
        positions: np.ndarray,
        batch_size: int = 1000,
        platform: str = "auto",
        sampling_step: int = 1,
    ) -> np.ndarray:
        """
        Computes the Mean-Squared Displacement (MSD) for a trajectory.

        Args:
            positions (np.ndarray): Array of coordinates with shape (n_frames, n_beads, 3).
            batch_size (int): Number of beads to process per batch. Balances RAM/VRAM usage.
            platform (str): 'auto', 'cpu', or 'gpu'. If 'auto', uses GPU if available.

        Returns:
            np.ndarray: Computed MSD array of shape (n_frames, n_beads).
        """
        # 1. Input Validation
        if not isinstance(positions, np.ndarray):
            raise TypeError("Input positions must be a numpy.ndarray.")
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                f"Expected positions shape (n_frames, n_beads, 3), got {positions.shape}"
            )

        platform = platform.lower()
        if platform not in ["auto", "cpu", "gpu"]:
            raise ValueError("Platform must be 'auto', 'cpu', or 'gpu'.")
        if not isinstance(sampling_step, int) or sampling_step < 1:
            raise ValueError("sampling_step must be a positive integer.")

        # 2. Platform Routing Logic
        use_gpu = Analyzer._check_platform(platform)

        positions = positions[::sampling_step, :, :]

        # 3. Execution
        logger.info(
            "Computing MSD on %s (Batch size: %d)...",
            "GPU" if use_gpu else "CPU",
            batch_size,
        )
        return Analyzer._msd_fft_batch(positions, batch_size, use_gpu=use_gpu)

    @staticmethod
    def compute_msd_average(
        positions: np.ndarray,
        bead_types: Optional[np.ndarray] = None,
        batch_size: int = 100,
        platform: str = "auto",
        sampling_step: int = 1,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]]:
        """
        Computes full-lag MSD averages without materializing the full bead-resolved
        MSD matrix for all beads at once.

        Args:
            positions: Coordinates with shape (n_frames, n_beads, 3). This may be a
                normal ndarray or a disk-backed np.memmap returned by Trajectory.xyz.
            bead_types: Optional 1-D labels with length n_beads. If provided, the
                return payload includes per-type MSD averages.
            batch_size: Number of beads to process per FFT batch.
            platform: 'auto', 'cpu', or 'gpu'.
            sampling_step: Optional stride along the frame axis.

        Returns:
            Dictionary with keys:
                'msd': 1-D average MSD over all beads.
                'type_msd': dict mapping bead type to 1-D average MSD.
                'type_counts': dict mapping bead type to bead count.
        """
        if not isinstance(positions, np.ndarray):
            raise TypeError("Input positions must be a numpy.ndarray.")
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                f"Expected positions shape (n_frames, n_beads, 3), got {positions.shape}"
            )
        if not isinstance(sampling_step, int) or sampling_step < 1:
            raise ValueError("sampling_step must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        platform = platform.lower()
        if platform not in ["auto", "cpu", "gpu"]:
            raise ValueError("Platform must be 'auto', 'cpu', or 'gpu'.")

        use_gpu = Analyzer._check_platform(platform)

        positions_sliced = positions[::sampling_step, :, :]
        n_frames, n_beads, _ = positions_sliced.shape

        if bead_types is not None:
            bead_types = np.asarray(bead_types)
            if bead_types.shape[0] != n_beads:
                raise ValueError(
                    f"bead_types length ({bead_types.shape[0]}) does not match "
                    f"number of beads ({n_beads})."
                )

        total_sum = np.zeros(n_frames, dtype=np.float64)
        total_count = 0
        type_sums: Dict[str, np.ndarray] = {}
        type_counts: Dict[str, int] = {}

        logger.info(
            "Computing averaged MSD on %s (Bead batch size: %d)...",
            "GPU" if use_gpu else "CPU",
            batch_size,
        )

        try:
            for start_idx in range(0, n_beads, batch_size):
                end_idx = min(start_idx + batch_size, n_beads)
                coords_batch = positions_sliced[:, start_idx:end_idx, :]
                msd_batch = Analyzer._msd_fft_batch(
                    coords_batch, batch_size, use_gpu=use_gpu, free_gpu_pool=False
                )

                total_sum += np.sum(msd_batch, axis=1, dtype=np.float64)
                total_count += msd_batch.shape[1]

                if bead_types is not None:
                    batch_types = bead_types[start_idx:end_idx]
                    for bead_type in np.unique(batch_types):
                        label = str(bead_type)
                        mask = batch_types == bead_type
                        if label not in type_sums:
                            type_sums[label] = np.zeros(n_frames, dtype=np.float64)
                            type_counts[label] = 0
                        type_sums[label] += np.sum(
                            msd_batch[:, mask], axis=1, dtype=np.float64
                        )
                        type_counts[label] += int(np.sum(mask))

                del msd_batch
        finally:
            if use_gpu and hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()

        type_msd = {
            label: type_sum / type_counts[label]
            for label, type_sum in type_sums.items()
        }

        return {
            "msd": total_sum / total_count,
            "type_msd": type_msd,
            "type_counts": type_counts,
        }

    @staticmethod
    def compute_anomalous_dynamics(
        lag_times: np.ndarray, msd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the anomalous exponent alpha(t) and apparent diffusivity D_app(t)
        simultaneously. Fully vectorized to support both 1D and 2D MSD arrays.

        Motivation:
        1. Slicing [1:] safely bypasses the t=0 singularity without breaking 2D shapes.
        2. Calculating alpha and D_app together avoids redundant log/gradient operations.

        Args:
            lag_times (np.ndarray): 1D array of time lags, shape (M,).
            msd (np.ndarray): 1D array (M,) or 2D array (M, B) of Mean Squared Displacements.

        Returns:
            alpha (np.ndarray): Exponent alpha(t), same shape as msd.
            D_app (np.ndarray): Apparent diffusivity D_app(t), same shape as msd.
        """
        # 1. Validation & Shape Consistency
        if lag_times.ndim != 1:
            raise ValueError("lag_times must be a 1D array.")
        if msd.shape[0] != lag_times.shape[0]:
            raise ValueError(
                f"Time axis mismatch: lag_times {lag_times.shape}, msd {msd.shape}"
            )

        # Initialize full arrays with NaNs (t=0 will remain NaN)
        alpha_full = np.full_like(msd, np.nan, dtype=np.float64)
        D_app_full = np.full_like(msd, np.nan, dtype=np.float64)

        # 2. Slice off t=0 to avoid log(0) = -inf
        t_valid = lag_times[1:]
        msd_valid = msd[1:]

        # Suppress warnings for any zero or negative values in specific beads
        with np.errstate(divide="ignore", invalid="ignore"):
            log_t = np.log(t_valid)
            log_msd = np.log(msd_valid)

        # 3. Calculate alpha(t) via gradient
        # Motivation: np.gradient accurately computes the derivative on irregularly
        # spaced grids (like log_t) along the time axis (axis=0).
        alpha_valid = np.gradient(log_msd, log_t, axis=0)

        # 4. Calculate D_app(t)
        # Motivation: Reshape t_valid for broadcasting if msd is a 2D matrix (M, B)
        if msd.ndim == 2:
            t_bcast = t_valid[:, np.newaxis]
        else:
            t_bcast = t_valid

        D_app_valid = msd_valid / (t_bcast**alpha_valid)

        # 5. Populate the full arrays (leaving index 0 as NaN)
        alpha_full[1:] = alpha_valid
        D_app_full[1:] = D_app_valid

        return alpha_full, D_app_full

    @staticmethod
    def collect_distances_from_center_unified(
            coords_trajectory: np.ndarray,
            bead_types: np.ndarray,
            sampling_rate: int = 1,
            batch_size: int = 1000,
            device: str = 'cpu',
            center: np.ndarray = None,
        ) -> Dict[str, np.ndarray]:
        R"""
        Unified version for collecting distances from a specified fixed center with batching support.
        
        Changes from COM version:
        - Calculates distance from a fixed coordinate (default: origin) instead of Center of Mass.
        - Removed bead_masses argument.
        
        Args:
            coords_trajectory: (n_frames, n_beads, 3) array of coordinates.
            bead_types: (n_beads,) array of bead types.
            sampling_rate: Interval for frame sampling.
            batch_size: Number of frames to process per batch.
            device: 'cpu' or 'gpu'.
            center: (n_dims,) array specifying the center coordinate. Defaults to origin.
            
        Returns:
            Dict mapping bead_type -> flattened array of distances.
        """
        # --- 1. Preparation ---
        coords_sliced = coords_trajectory[::sampling_rate]
        
        # Validate shape and get dimensions
        if coords_sliced.ndim != 3:
            raise ValueError(f"coords_trajectory must be 3D (frames, beads, dims), got {coords_sliced.ndim}D")
        
        n_frames, n_beads, n_dims = coords_sliced.shape
        
        # Select backend
        use_gpu = device.lower() == 'gpu' and CUPY_AVAILABLE
        if device.lower() == 'gpu' and not CUPY_AVAILABLE:
            logger.warning("GPU requested but CuPy is not available. Falling back to CPU.")
        xp = cp if use_gpu else np

        # Handle Center Coordinate
        if center is None:
            center = np.zeros(n_dims)
        else:
            center = np.asarray(center)
            # Ensure center matches the spatial dimension of the trajectory
            if center.shape[-1] != n_dims:
                raise ValueError(
                    f"Center dimension ({center.shape[-1]}) does not match "
                    f"trajectory spatial dimension ({n_dims})"
                )
        
        # Move center to correct device (cpu/gpu)
        center_xp = xp.asarray(center, dtype=coords_sliced.dtype)
        
        # Prepare bead type indices (always on CPU for dictionary keys)
        bead_types = np.asarray(bead_types)
        if bead_types.shape[0] != coords_sliced.shape[1]:
            raise ValueError(f"bead_types length ({bead_types.shape[0]}) must match "
                        f"number of beads in trajectory ({coords_sliced.shape[1]})")

        unique_types = np.unique(bead_types)
        type_indices = {t: np.where(bead_types == t)[0] for t in unique_types}
        
        # accumulator
        distances_by_type_accumulator = defaultdict(list)

        logger.info(
            "Starting %s analysis (Fixed Center) with batch_size=%d...",
            "GPU" if use_gpu else "CPU",
            batch_size,
        )
        logger.info("Center coordinate: %s (dims=%d)", center, n_dims)

        # --- 2. Batch Processing Loop ---
        try:
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                
                # Load batch to device
                batch_coords = xp.asarray(coords_sliced[start:end])

                # A. Relative Coordinates using broadcasting
                # batch_coords shape: (batch_size, n_beads, n_dims)
                # center_xp shape: (n_dims,)
                # Result shape: (batch_size, n_beads, n_dims)
                centered_batch = batch_coords - center_xp

                # B. L2 Norm along the last axis (spatial dimension)
                # result shape: (batch_size, n_beads)
                dist_batch = xp.linalg.norm(centered_batch, axis=-1)

                # C. Collect data
                for btype, indices in type_indices.items():
                    # Select distances for specific bead types and flatten
                    selected_dist = dist_batch[:, indices].ravel()
                    
                    if use_gpu:
                        distances_by_type_accumulator[btype].append(cp.asnumpy(selected_dist))
                    else:
                        distances_by_type_accumulator[btype].append(selected_dist)

                if use_gpu:
                    del batch_coords, centered_batch, dist_batch
        finally:
            if use_gpu and hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()

        # --- 3. Final Consolidation ---
        final_distributions = {}
        for btype, list_of_arrays in distances_by_type_accumulator.items():
            final_distributions[btype] = np.concatenate(list_of_arrays)

        return final_distributions

    @staticmethod
    def collect_distances_from_com_unified(
        coords_trajectory: np.ndarray,
        bead_types: np.ndarray,
        bead_masses: Optional[np.ndarray] = None,
        sampling_rate: int = 1,
        batch_size: int = 1000,
        device: str = 'cpu'
    ) -> Dict[str, np.ndarray]:
        R"""
        Unified version for collecting distances from COM with batching support for CPU/GPU.
        """
        # --- 1. Preparation ---
        coords_sliced = coords_trajectory[::sampling_rate]
        if coords_sliced.size == 0:
            raise ValueError(f"coords_trajectory is empty after slicing with sampling_rate={sampling_rate}")

        n_frames, n_beads, _ = coords_sliced.shape
        use_gpu = device.lower() == 'gpu' and CUPY_AVAILABLE
        if device.lower() == 'gpu' and not CUPY_AVAILABLE:
            logger.warning("GPU requested but CuPy is not available. Falling back to CPU.")
        xp = cp if use_gpu else np

        if len(bead_types) != n_beads:
            raise ValueError(
                f"Dimension mismatch: bead_types length ({len(bead_types)}) "
                f"does not match trajectory n_beads ({n_beads})"
            )

        if bead_masses is None:
            masses = xp.ones(n_beads)
        else:
            masses = xp.asarray(bead_masses)

        bead_types = np.asarray(bead_types)
        if bead_types.shape[0] != coords_sliced.shape[1]:
            raise ValueError(f"bead_types length ({bead_types.shape[0]}) must match "
                        f"number of beads in trajectory ({coords_sliced.shape[1]})")

        unique_types = np.unique(bead_types)
        type_indices = {t: np.where(bead_types == t)[0] for t in unique_types}
        
        # accumulator
        distances_by_type_accumulator = defaultdict(list)

        logger.info(
            "Starting %s analysis with batch_size=%d...",
            "GPU" if use_gpu else "CPU",
            batch_size,
        )

        # --- 2. Batch Processing Loop ---
        try:
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                # load batch to device
                batch_coords = xp.asarray(coords_sliced[start:end])

                # A. Calculate COM for each batch
                # result shape: (batch_size, 3)
                com_batch = xp.average(batch_coords, axis=1, weights=masses)

                # B. Relative Coordinates using broadcasting
                # Motivation: com_batch[:, None, :] using broadcasting: (N, 3) -> (N, 1, 3)
                centered_batch = batch_coords - com_batch[:, xp.newaxis, :]

                # C. L2 Norm
                # result shape: (batch_size, n_beads)
                dist_batch = xp.linalg.norm(centered_batch, axis=2)

                # D. collect data
                for btype, indices in type_indices.items():
                    selected_dist = dist_batch[:, indices].ravel()
                    if use_gpu:
                        distances_by_type_accumulator[btype].append(cp.asnumpy(selected_dist))
                    else:
                        distances_by_type_accumulator[btype].append(selected_dist)

                if use_gpu:
                    del batch_coords, com_batch, centered_batch, dist_batch
        finally:
            if use_gpu and hasattr(cp, "get_default_memory_pool"):
                cp.get_default_memory_pool().free_all_blocks()

        # --- 3. Final Consolidation ---
        final_distributions = {}
        for btype, list_of_arrays in distances_by_type_accumulator.items():
            final_distributions[btype] = np.concatenate(list_of_arrays)

        return final_distributions


def _sorted_frame_keys(h5_file):
    """Return original integer frame keys in numerical order, excluding metadata."""
    keys = []
    for key in h5_file.keys():
        try:
            int(key)
        except ValueError:
            continue
        keys.append(key)
    return sorted(keys, key=int)


def _orthorhombic_lengths(boxes):
    """Validate the box convention shared by trajectory and Hi-C MIC paths."""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.shape[-2:] != (3, 3) or not np.all(np.isfinite(boxes)):
        raise ValueError("Box vectors must be finite 3x3 matrices")
    lengths = np.diagonal(boxes, axis1=-2, axis2=-1)
    diagonal = np.zeros_like(boxes)
    diagonal[..., np.arange(3), np.arange(3)] = lengths
    if np.any(lengths <= 0) or not np.allclose(boxes, diagonal, rtol=0, atol=1e-12):
        raise ValueError("MIC requires orthorhombic boxes with positive lengths")
    return lengths


class TrajectoryLoader:
    """
    Loader for HDF5 trajectories .
    """

    @staticmethod
    def load(traj_file: Union[str, Path], d: int = 1) -> np.ndarray:
        """
        Load trajectory from an HDF5 file.

        Parameters
        ----------
        traj_file : str or Path
            Path to the HDF5 trajectory file.

        Returns
        -------
        np.ndarray
            Array of positions with shape (T, N, 3) for T frames.
        """
        if not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError("d must be a positive integer frame-ID divisor")
        pos: list[np.ndarray] = []
        with h5py.File(str(traj_file), "r") as f:
            for key in _sorted_frame_keys(f):
                if int(key) % d == 0:
                    pos.append(np.array(f[key]))

        return np.array(pos)


# For using as independent functions
# self should be the object of the class Trajectory


class Trajectory:
    """
    Trajectory class for processing cndb/xyz/pdb formats.
    """

    def __init__(self, filename: str = None):
        # initialize attributes (Snake Case applied)
        self.cndb = None
        self.filename = filename
        self.n_beads = 0  # renamed from Nbeads
        self.n_frames = 0  # renamed from Nframes
        self.frame_keys = []
        self.chrom_seq = []
        self.unique_chrom_seq = set()
        self.dict_chrom_seq = {}
        self.topology = None
        self.box_vectors = None

        # if filename is provided, load the trajectory
        if filename:
            self.load(filename)

    def load(self, filename: str):
        """
        Loads cndb file, including types, topology, and PBC box vectors.
        """
        self.filename = filename
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        self.close()
        self.cndb = h5py.File(filename, "r")

        # Sort frame keys
        self.frame_keys = _sorted_frame_keys(self.cndb)
        frame_keys = self.frame_keys
        self.n_frames = len(frame_keys)

        if self.n_frames == 0:
            logger.warning("No frames found in file.")
            return self

        # --- 1. Load Bead Number ---
        first_frame_data = self.cndb[frame_keys[0]]
        self.n_beads = first_frame_data.shape[0]

        # --- 2. Load Types ---
        if "types" in self.cndb:
            raw_types = self.cndb["types"]
            self.chrom_seq = [
                t.decode("utf-8") if isinstance(t, bytes) else t for t in raw_types
            ]
        else:
            logger.warning("'types' dataset not found. Assuming uniform bead types.")
            self.chrom_seq = ["U"] * self.n_beads

        self.unique_chrom_seq = set(self.chrom_seq)
        self.dict_chrom_seq = {
            tt: [i for i, e in enumerate(self.chrom_seq) if e == tt]
            for tt in self.unique_chrom_seq
        }

        # --- 3. Load Native Topology ---
        self.topology = None
        if "topology" in self.cndb:
            try:
                self.topology = self._load_topology_from_h5(self.cndb)
            except Exception as e:
                logger.warning(f"Failed to load topology data from HDF5: {e}")

        # --- 4. Load Box Vectors ---
        if "box" in first_frame_data.attrs:
            self.box_vectors = np.zeros((self.n_frames, 3, 3))
            for i, key in enumerate(frame_keys):
                if "box" in self.cndb[key].attrs:
                    self.box_vectors[i] = self.cndb[key].attrs["box"]
                elif i > 0:
                    self.box_vectors[i] = self.box_vectors[i - 1]
        else:
            self.box_vectors = None

        logger.info(f"Loaded {self.filename}: {self.n_frames} frames, {self.n_beads} beads.")
        if self.topology:
            logger.info(
                f"Topology loaded: {self.topology.getNumAtoms()} atoms, {self.topology.getNumBonds()} bonds"
            )
        if self.box_vectors is not None:
            logger.info(f"Box vectors loaded. Shape: {self.box_vectors.shape}")

        return self

    def xyz(
        self,
        frames=[0, None, 1],
        bead_selection=None,
        xyz_cols=[0, 1, 2],
        frame_batch_size: Optional[int] = None,
        dtype=None,
        storage: str = "memory",
        memmap_path: Optional[str] = None,
    ):
        """
        Get the selected beads' 3D position from a **cndb** file for multiple frames.

        Args:
            frames: [start, end, step] ordinal positions in numerically sorted frames.
            bead_selection: Optional bead indices to retrieve.
            xyz_cols: Coordinate columns to retrieve.
            frame_batch_size: Optional number of frames to materialize per read
                batch. The final return value remains a single array.
            dtype: Optional output dtype, e.g. np.float32 to reduce memory.
            storage: 'memory' for a normal ndarray or 'memmap' for a disk-backed
                array.
            memmap_path: Required output path when storage='memmap'.
        """
        if self.cndb is None:
            raise RuntimeError("No file loaded. Call load() first.")

        storage = storage.lower()
        if storage not in {"memory", "memmap"}:
            raise ValueError("storage must be 'memory' or 'memmap'.")
        if frame_batch_size is not None and frame_batch_size < 1:
            raise ValueError("frame_batch_size must be a positive integer.")

        if bead_selection is None:
            selection = np.arange(self.n_beads)
        else:
            selection = np.array(bead_selection)

        frame_indices = self._frame_positions(frames)
        if not frame_indices:
            return np.empty((0, len(selection), len(xyz_cols)), dtype=dtype)

        first_frame = np.array(self.cndb[self.frame_keys[frame_indices[0]]])
        first_selected = np.take(
            np.take(first_frame, selection, axis=0), xyz_cols, axis=1
        )
        if dtype is not None:
            first_selected = first_selected.astype(dtype, copy=False)

        out_shape = (len(frame_indices),) + first_selected.shape
        out_dtype = first_selected.dtype

        if storage == "memmap":
            if memmap_path is None:
                raise ValueError("memmap_path is required when storage='memmap'.")
            output = np.memmap(memmap_path, dtype=out_dtype, mode="w+", shape=out_shape)
        elif frame_batch_size is not None or dtype is not None:
            output = np.empty(out_shape, dtype=out_dtype)
        else:
            return np.array([
                np.take(np.take(self.cndb[self.frame_keys[i]][...], selection, axis=0), xyz_cols, axis=1)
                for i in frame_indices
            ])

        batch_size = frame_batch_size or len(frame_indices)
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[batch_start:batch_start + batch_size]
            for j, i in enumerate(batch_indices):
                frame_data = self.cndb[self.frame_keys[i]][...]
                output[batch_start + j] = np.take(
                    np.take(frame_data, selection, axis=0), xyz_cols, axis=1
                )

        if isinstance(output, np.memmap):
            output.flush()
        return output

    def _frame_positions(self, frames):
        """Select ordinal frame positions, independently of stored frame IDs."""
        start, end, step = frames
        if not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("Frame stride must be a positive integer")
        start = max(0, start)
        end = self.n_frames if end is None else min(end, self.n_frames)
        return range(start, end, step)

    def close(self):
        """Close the HDF5 file handle."""
        if hasattr(self, "cndb") and self.cndb:
            self.cndb.close()
        self.cndb = None

    def __del__(self):
        self.close()

    @staticmethod
    def _load_topology_from_h5(h5_file):
        """
        Reconstructs an OpenMM Topology object from HDF5 datasets.
        """

        if "topology" not in h5_file:
            return None

        grp = h5_file["topology"]
        atoms_arr = grp["atoms"][:]
        chain_ids = grp["chain_ids"][:]
        res_names = grp["res_names"][:]
        bonds_arr = grp["bonds"][:] if "bonds" in grp else []

        new_top = Topology()

        # 1. create Chains
        created_chains = [
            new_top.addChain(cid.decode("utf-8") if isinstance(cid, bytes) else cid)
            for cid in chain_ids
        ]

        # 2. create Residues and Atoms
        res_objs = {}
        atom_objs = []
        for i, atom_data in enumerate(atoms_arr):
            name = atom_data["name"].decode("utf-8")
            elem_sym = atom_data["element"].decode("utf-8")
            res_idx = atom_data["res_idx"]
            chain_idx = atom_data["chain_idx"]

            # create Residue
            if res_idx not in res_objs:
                rname = res_names[res_idx]
                rname_str = rname.decode("utf-8") if isinstance(rname, bytes) else rname
                res_objs[res_idx] = new_top.addResidue(
                    rname_str, created_chains[chain_idx]
                )

            # create Atom
            try:
                elem_obj = Element.getBySymbol(elem_sym)
            except KeyError:
                elem_obj = None
            atom_objs.append(new_top.addAtom(name, elem_obj, res_objs[res_idx]))

        # 3. create Bonds
        for idx1, idx2 in bonds_arr:
            new_top.addBond(atom_objs[idx1], atom_objs[idx2])

        return new_top

    @property
    def chain_info(self):
        """
        Returns a summary list of tuples: [(ChainID, NumAtoms), ...]
        """
        if self.topology is None:
            return []
        info = []
        for chain in self.topology.chains():
            n_atoms = sum(1 for _ in chain.atoms())
            info.append((chain.id, n_atoms))
        return info

    # As requested, this function is added to the Trajectory class
    def compute_rg_type(
        self, get_components: bool = False, custom_types: Optional[List] = None
    ):
        """
        Function to compute Radius of Gyration (Rg) classified by particle type.

        Parameters:
            get_components (bool): If True, returns both total Rg and its XYZ components.
                                Default is False.
            custom_types (list, optional): A list of custom bead types. If provided,
                                this will be used instead of self.chrom_seq.

        Returns:
            results (dict): A dictionary containing Rg data.
                - If get_components is False:
                    key: 'general' and each type name (e.g., 'A', 'B')
                    value: Corresponding Rg numpy array of shape (T,)
                - If get_components is True:
                    key: 'general' and each type name
                    value: A dictionary with:
                        - 'total': np.ndarray of shape (T,)
                        - 'components': np.ndarray of shape (T, 3)
        """

        # 1. Get coordinates and sequence from SELF
        # Dimension: (T, N, 3)
        all_positions = np.asarray(self.xyz(frames=[0, None, 1], bead_selection=None))

        if custom_types is not None:
            logger.info("Notice: Using custom-defined bead types for Rg calculation.")
            bead_types = np.asarray(custom_types)
        else:
            bead_types = np.asarray(self.chrom_seq)

        if bead_types.shape[0] != all_positions.shape[1]:
            raise ValueError(
                f"bead_types length ({bead_types.shape[0]}) must match "
                f"number of beads in trajectory ({all_positions.shape[1]})"
            )

        # 2. Initialize result dictionary
        results = {}

        # 3. Calculate 'general' Rg (all beads)
        # Forward the get_components flag to the static method
        results["general"] = Analyzer.compute_RG(
            all_positions, return_components=get_components
        )

        # 4. Check if system is Heterogeneous
        unique_types = np.unique(bead_types)

        # If type count is greater than 1, calculate by type
        if len(unique_types) > 1:
            for t_type in unique_types:
                # Create Boolean Mask
                mask = bead_types == t_type

                # Slice positions: [all frames, filtered beads, xyz]
                subset_positions = all_positions[:, mask, :]

                # Ensure type name is string format as key
                key_name = str(t_type)

                # 5. Calculate Rg for this type
                # The return type of Analyzer.compute_RG depends on get_components
                rg_data = Analyzer.compute_RG(
                    subset_positions, return_components=get_components
                )

                # If get_components is True, rg_data is a tuple (total, xyz)
                # We can store it as a sub-dictionary for better readability
                if get_components:
                    total_rg, xyz_rg = rg_data
                    results[key_name] = {"total": total_rg, "components": xyz_rg}
                    # Also update "general" to a dict format for structure consistency
                    if key_name == str(
                        unique_types[0]
                    ):  # Only need to format "general" once
                        gen_total, gen_xyz = results["general"]
                        results["general"] = {"total": gen_total, "components": gen_xyz}
                else:
                    results[key_name] = rg_data

        return results

    def compute_persistent_length_type(
        self,
        custom_types: Optional[List] = None,
        max_bond_sep: Optional[int] = None,
        boundary_mode: str = 'segment',
        device: str = 'gpu',
        use_pbc: bool = True,
        positions: Optional[np.ndarray] = None,
        box_vectors: Optional[np.ndarray] = None,
    ):
        R"""
        Compute persistent length (Lp) classified by particle type.

        Follows the same pattern as ``compute_rg_type``.  Returns the tangent
        correlation function C(s) and the fitted Lp for the whole chain
        (``'general'``) and for each bead-type separately.

        Persistent length is extracted from the tangent-tangent correlation:
            C(s) = <û_i · û_{i+s}>  ≈  exp(-s·b / Lp)
        where b is the mean bond length.

        Parameters
        ----------
        custom_types : list, optional
            A list of custom bead types.  If provided, used instead of
            ``self.chrom_seq``.
        max_bond_sep : int, optional
            Maximum bond-index separation *s* to compute.  ``None`` lets the
            ``Analyzer`` auto-select using dual-bound logic:

            - **Upper bound**: ``int(n_bonds * 0.5)`` — caps at half the
              segment length so each C(s) is averaged over enough pairs.
            - **Lower bound**: ``10`` data points minimum for a reliable
              exponential fit.

            When these bounds are incompatible (segment too short), C(s) is
            still computed but ``fit_reliable`` is ``False`` and Lp may be
            unreliable.
        device : str, default ``'gpu'``
            Compute backend passed to ``Analyzer.compute_tangent_correlation``.
            ``'gpu'`` uses CuPy for acceleration (requires CuPy).
        use_pbc : bool, default ``True``
            If ``True`` and box vectors are available, apply the Minimum Image
            Convention (MIC) when calculating bond vectors.
        positions : np.ndarray, optional
            Pre-loaded positions array of shape ``(T, N, 3)``. If ``None``, calls
            ``self.xyz()``. Useful for providing concatenated coordinates from
            multiple replicas.
        box_vectors : np.ndarray, optional
            Pre-loaded box vectors. If ``None``, uses ``self.box_vectors``. Useful
            when providing concatenated coordinates.
        boundary_mode : str, default ``'segment'``
            How to handle type boundaries when computing per-type Lp.

            ``'segment'`` (default)
                For each type, find *contiguous* segments of that type along
                the chain.  Tangent correlations are computed strictly within
                each segment.  Bonds at the segment boundaries are included.

            ``'exclude_boundary'``
                Same as ``'segment'``, but additionally **removes** the first
                and last bond of each segment if the segment does not sit at
                the chain terminus.  This avoids any influence from the
                junction between different types.

            ``'include_boundary'``
                A bond is assigned to type X if **at least one** of its two
                endpoint beads has type X.  Tangent correlations for type X
                are then computed only between bonds that are both assigned to
                X **and** form a contiguous sub-sequence.

        Returns
        -------
        results : dict
            Keys are ``'general'`` and each unique type name (e.g. ``'A'``,
            ``'B'``).  Each value is a dict containing:

            - ``'tangent_corr'``    : np.ndarray, shape (max_s+1,) — C(s)
            - ``'contour_length'``  : np.ndarray, shape (max_s+1,) — s × ⟨b⟩
            - ``'Lp'``              : float — fitted persistent length
            - ``'Lp_se'``           : float — fit-based standard error of Lp
            - ``'Lp_ci95'``         : tuple — approximate 95% confidence interval
            - ``'mean_bond_length'``: float — average bond length for that subset
            - ``'fit_reliable'``    : bool — whether enough points for robust fit
            - ``'r_squared'``       : float — R^2 value of the fit
            - ``'fit_slope'``       : float — zero-intercept log-linear slope
            - ``'fit_slope_se'``    : float — standard error of ``fit_slope``
            - ``'fit_cov'``         : np.ndarray — covariance matrix for slope
            - ``'fit_n_points'``    : int — number of points used in the fit
            - ``'fit_details'``     : str — details of fit reliability

        Notes
        -----
        For heterogeneous chains (e.g. A-B copolymer), the *general* result
        uses ALL bonds regardless of type, while each type entry only uses
        the bonds *within* contiguous segments of that type (subject to
        ``boundary_mode``).  Segments with fewer than ``2 * MIN_FIT_POINTS``
        bonds (= 20 by default, after boundary trimming) are skipped since
        the dual-bound constraints cannot both be satisfied.
        """

        # ============================================================
        # 1. Load coordinates & types
        # ============================================================
        if positions is not None:
            all_positions = np.asarray(positions)
        else:
            all_positions = np.asarray(
                self.xyz(frames=[0, None, 1], bead_selection=None)
            )  # (T, N, 3)

        source_box = box_vectors if box_vectors is not None else self.box_vectors
        box_to_use = source_box if use_pbc else None

        if custom_types is not None:
            logger.info("Notice: Using custom-defined bead types for Lp calculation.")
            bead_types = np.asarray(custom_types)
        else:
            bead_types = np.asarray(self.chrom_seq)

        n_beads = all_positions.shape[1]
        if bead_types.shape[0] != n_beads:
            raise ValueError(
                f"bead_types length ({bead_types.shape[0]}) must match "
                f"number of beads in trajectory ({n_beads})"
            )

        # ============================================================
        # 2. 'general' — whole chain
        # ============================================================
        results = {}
        results['general'] = Analyzer.compute_tangent_correlation(
            all_positions,
            max_s=max_bond_sep,
            device=device,
            box_vectors=box_to_use,
        )

        # ============================================================
        # 3. Per-type computation
        # ============================================================
        unique_types = np.unique(bead_types)

        if len(unique_types) <= 1:
            # Homogeneous chain — nothing extra to compute
            return results

        for t_type in unique_types:
            key_name = str(t_type)

            # --- Determine bond indices for this type ---
            segments = self._find_contiguous_segments(bead_types, t_type)

            # Convert bead-level segments to bond-level index lists
            bond_index_lists = []
            for seg_start, seg_end in segments:
                # Bonds in this segment: indices [seg_start, seg_end - 2]
                # (bond i connects bead i to bead i+1)
                seg_n_beads = seg_end - seg_start
                if seg_n_beads < 2:
                    continue  # need at least 2 beads for 1 bond

                b_start = seg_start
                b_end = seg_end - 1  # exclusive, bond indices

                # --- Apply boundary_mode trimming ---
                if boundary_mode == 'exclude_boundary':
                    # Trim first bond if not at chain start
                    if seg_start > 0:
                        b_start += 1
                    # Trim last bond if not at chain end
                    if seg_end < n_beads:
                        b_end -= 1
                elif boundary_mode == 'include_boundary':
                    # Extend to include bonds at boundaries (one endpoint matches)
                    # Left boundary: bond (seg_start - 1) if exists
                    if seg_start > 0:
                        b_start -= 1
                    # Right boundary: bond (seg_end - 1) if exists and < n_beads - 1
                    if seg_end < n_beads:
                        b_end += 1
                elif boundary_mode != 'segment':
                    raise ValueError(
                        f"Unknown boundary_mode '{boundary_mode}'. "
                        f"Choose from 'segment', 'exclude_boundary', 'include_boundary'."
                    )

                if b_end > b_start:
                    bond_index_lists.append(np.arange(b_start, b_end))

            if not bond_index_lists:
                # No valid segments for this type
                results[key_name] = {
                    'tangent_corr': np.array([1.0]),
                    'contour_length': np.array([0.0]),
                    'mean_bond_length': np.nan,
                    'fit_reliable': False,
                    'fit_details': "No valid segments.",
                    **Analyzer._empty_persistent_length_fit(),
                }
                continue

            # --- Compute tangent correlation per segment, then aggregate ---
            # Strategy: for each contiguous segment, compute C(s) using the
            #           static method with bond_indices, then weight-average
            #           across segments by the number of contributing pairs.
            #
            # Segment skip criterion:
            #   A segment with n_bonds bonds yields max_s = n_bonds * 0.5.
            #   For fit_reliable, we need max_s >= MIN_FIT_POINTS (=10).
            #   Therefore: n_bonds >= 2 * MIN_FIT_POINTS (=20).
            #   Shorter segments are skipped to avoid unreliable Lp.
            MIN_SEG_BONDS = 20  # = 2 * MIN_FIT_POINTS used in Analyzer
            seg_results = []
            seg_weights = []  # weight = number of bonds in segment

            for bond_idx in bond_index_lists:
                seg_len = len(bond_idx)
                if seg_len < MIN_SEG_BONDS:
                    continue  # too short for both noise and fit constraints

                seg_res = Analyzer.compute_tangent_correlation(
                    all_positions,
                    max_s=max_bond_sep,
                    bond_indices=bond_idx,
                    device=device,
                    box_vectors=box_to_use,
                )
                seg_results.append(seg_res)
                seg_weights.append(seg_len)

            if not seg_results:
                results[key_name] = {
                    'tangent_corr': np.array([1.0]),
                    'contour_length': np.array([0.0]),
                    'mean_bond_length': np.nan,
                    'fit_reliable': False,
                    'fit_details': "All segments too short.",
                    **Analyzer._empty_persistent_length_fit(),
                }
                continue

            # Weighted average of tangent correlation across segments
            # Pad shorter C(s) arrays with NaN, then do weighted nanmean
            max_len = max(len(r['tangent_corr']) for r in seg_results)
            weights_arr = np.array(seg_weights, dtype=float)

            combined_corr = np.full((len(seg_results), max_len), np.nan)
            for idx, r in enumerate(seg_results):
                L = len(r['tangent_corr'])
                combined_corr[idx, :L] = r['tangent_corr']

            # Weighted nanmean along axis 0
            # For each s, average only over segments that have a value
            avg_corr = np.zeros(max_len)
            for s_idx in range(max_len):
                valid_mask = ~np.isnan(combined_corr[:, s_idx])
                if np.any(valid_mask):
                    w = weights_arr[valid_mask]
                    avg_corr[s_idx] = np.average(
                        combined_corr[valid_mask, s_idx], weights=w
                    )
                else:
                    avg_corr[s_idx] = np.nan

            # Weighted-average mean bond length
            mean_bl = np.average(
                [r['mean_bond_length'] for r in seg_results],
                weights=weights_arr,
            )

            # Re-fit Lp from the aggregated C(s)
            s_vals = np.arange(max_len)
            contour_length = s_vals * mean_bl

            # Assess reliability based on aggregated correlation
            MIN_FIT_POINTS = 10
            fit_reliable = (max_len - 1 >= MIN_FIT_POINTS) # max_len is max_s + 1
            fit_details = ""
            if not fit_reliable:
                fit_details = f"Aggregated max_s={max_len - 1} < {MIN_FIT_POINTS}. "

            if max_len > 1 and avg_corr[1] <= 0:
                fit_reliable = False
                fit_details += (
                    "C(1) <= 0: Chain exhibits local anti-correlation or strong "
                    "flexibility. WLC model is inapplicable."
                )

            fit_result = Analyzer._fit_persistent_length_from_corr(
                avg_corr,
                mean_bond_length=mean_bl,
                min_fit_points=MIN_FIT_POINTS,
            )

            if fit_result['fit_n_points'] < 2:
                if fit_reliable:
                    fit_reliable = False
                    fit_details += "Not enough positive C(s) data points for fitting."
            elif not np.isfinite(fit_result['Lp']):
                if fit_reliable:
                    fit_reliable = False
                    fit_details += "Persistent length fit produced a non-finite result."
            elif fit_result['fit_slope'] >= 0:
                if fit_reliable:
                    fit_reliable = False
                    fit_details += "Non-decaying correlation (slope >= 0)."
            elif fit_reliable and fit_result['r_squared'] < 0.8:
                fit_reliable = False
                fit_details += f"Poor exponential fit (R^2 = {fit_result['r_squared']:.2f})."

            # Aggregated fit_reliable: also check if any individual segment failed to be reliable
            all_reliable = all(r['fit_reliable'] for r in seg_results)
            if fit_reliable and not all_reliable:
                fit_reliable = False
                fit_details += "Some individual segments were unreliable."

            if fit_reliable and not fit_details:
                fit_details = "Fit is reliable."

            results[key_name] = {
                'tangent_corr': avg_corr,
                'contour_length': contour_length,
                'mean_bond_length': mean_bl,
                'fit_reliable': fit_reliable,
                'fit_details': fit_details.strip(),
                **fit_result,
            }

        return results

    @staticmethod
    def _find_contiguous_segments(
        bead_types: np.ndarray, target_type
    ) -> List[Tuple[int, int]]:
        """
        Find contiguous runs of *target_type* in a 1-D type array.

        Parameters
        ----------
        bead_types : np.ndarray
            1-D array of bead type labels.
        target_type : str or comparable
            The type to search for.

        Returns
        -------
        list of (start, end)
            Each tuple gives the **bead** index range [start, end) of a
            contiguous segment of *target_type*.
        """
        mask = (bead_types == target_type)
        segments = []
        in_seg = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_seg:
                start = i
                in_seg = True
            elif not val and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, len(mask)))
        return segments

    def xyz_wrapped(self, frames=[0, None, 1], bead_selection=None, xyz_cols=[0, 1, 2]):
        """
        Get the *WRAPPED* coordinates (inside the simulation box) for selected beads.

        This acts as a wrapper around self.xyz() but applies the periodic wrapping
        operation using self.box_vectors.

        Args:
            frames (list): [start, end, step]
            bead_selection (list/array): Indices of beads to retrieve.
            xyz_cols (list): Indices of dimensions to retrieve.

        Returns:
            np.ndarray: Wrapped coordinates with shape (T, N, 3).
        """
        # 1. Get raw Unwrapped coordinates (T, N, 3)
        # Directly reuse existing xyz function
        coords_unwrapped = self.xyz(
            frames=frames, bead_selection=bead_selection, xyz_cols=xyz_cols
        )

        # 2. Get and process Box Vectors
        if self.box_vectors is None:
            logger.warning("No box vectors found. Returning unwrapped coordinates.")
            return coords_unwrapped

        subset_boxes = self.box_vectors[list(self._frame_positions(frames))]
        box_lengths = _orthorhombic_lengths(subset_boxes)[:, np.asarray(xyz_cols)]
        return coords_unwrapped % box_lengths[:, np.newaxis, :]

    def check_if_wrapped(self):
        """
        Check if the trajectory contains breaks due to Periodic Boundary Conditions (PBC).
        Principle: Calculate distances between adjacent beads. If distances are close to the box size, wrapping has occurred.
        """
        # Get coordinates of the first frame (N, 3)
        coords = self.xyz(frames=[0, 1, 1])[0]

        # Calculate adjacent bead distances: ||r_{i+1} - r_i||
        diffs = coords[1:] - coords[:-1]
        dists = np.linalg.norm(diffs, axis=1)

        logger.info(f"Max bond distance: {np.max(dists):.4f}")
        logger.info(f"Mean bond distance: {np.mean(dists):.4f}")

        # Assume normal bond length is around 1.0. Large values indicate the trajectory is wrapped.
        if np.max(dists) > 10.0:  # Threshold set to, e.g., 10 times the bond length
            logger.warning(
                "Extremely large bond distance detected! Data appears to be Wrapped."
            )
            logger.warning("Direct Rg calculation will be incorrect! Must Unwrap first.")
        else:
            logger.info(
                "Max bond distance is normal. Data appears to be Unwrapped (continuous) or the system has not crossed boundaries."
            )

    def get_distance_distribution_by_separation_and_type(
        self, 
        s: int = 1, 
        bead_types: Optional[np.ndarray] = None,
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        sampling_step: int = 1,
        batch_size: int = 1000,
        wrapped: bool = False,
        device: str = 'gpu'  
    ) -> Dict[str, np.ndarray]:
        """Distances at contour separation s, within each topology chain.

        Frame bounds select ordinal positions once. With wrapped=True, distances
        use the minimum image of the selected orthorhombic boxes. Without native
        topology the particle array is treated as one linear chain; chain order
        is also used for rings (no shortest-ring-distance convention is implied).
        """
        frames = [start_frame, end_frame, sampling_step]
        if device.lower() not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        xyz = self.xyz(frames=frames)
        bead_types = np.asarray(self.chrom_seq if bead_types is None else bead_types)
        if bead_types.dtype.kind == 'S':
            bead_types = np.char.decode(bead_types, 'utf-8')
        if bead_types.shape != (self.n_beads,):
            raise ValueError("bead_types must have one label per particle")
        if not isinstance(s, (int, np.integer)) or not 0 < s < self.n_beads:
            raise ValueError(f"Separation s must be between 1 and {self.n_beads-1}")
        boxes = None
        if wrapped:
            if self.box_vectors is None:
                raise ValueError("wrapped distances require periodic box vectors")
            boxes = _orthorhombic_lengths(self.box_vectors[list(self._frame_positions(frames))])
        if not len(xyz):
            return {}
        if self.topology is None:
            chains = [np.arange(self.n_beads)]
        else:
            if self.topology.getNumAtoms() != self.n_beads:
                raise ValueError("Topology particle count must match trajectory")
            chains = [np.array([a.index for a in c.atoms()]) for c in self.topology.chains()]
        pairs = [(chain[:-s], chain[s:]) for chain in chains if len(chain) > s]
        if not pairs:
            return {}
        i = np.concatenate([pair[0] for pair in pairs])
        j = np.concatenate([pair[1] for pair in pairs])
        xp = cp if device.lower() == 'gpu' and CUPY_AVAILABLE else np
        return self._calc_distances_worker(xyz, bead_types, i, j, batch_size, boxes, xp)

    @staticmethod
    def _calc_distances_worker(xyz, bead_types, i, j, batch_size, boxes, xp):
        # One pair/MIC implementation for both NumPy and CuPy backends.
        unique_types = np.unique(bead_types)
        masks = {}
        for t1, t2 in combinations_with_replacement(unique_types, 2):
            mask = ((bead_types[i] == t1) & (bead_types[j] == t2))
            if t1 != t2:
                mask |= ((bead_types[i] == t2) & (bead_types[j] == t1))
            if np.any(mask):
                masks[f"{t1}-{t2}"] = xp.asarray(mask)
        results = {key: [] for key in masks}
        idx_i, idx_j = xp.asarray(i), xp.asarray(j)
        for start in range(0, len(xyz), batch_size):
            stop = min(start + batch_size, len(xyz))
            coords = xp.asarray(xyz[start:stop])
            dr = coords[:, idx_j, :] - coords[:, idx_i, :]
            if boxes is not None:
                lengths = xp.asarray(boxes[start:stop])[:, None, :]
                dr -= lengths * xp.rint(dr / lengths)
            distances = xp.linalg.norm(dr, axis=2)
            for key, mask in masks.items():
                values = distances[:, mask].ravel()
                results[key].append(values if xp is np else cp.asnumpy(values))
        return {key: np.concatenate(values) for key, values in results.items()}


# --- External Wrappers (For Backward Compatibility / Functional Style) ---
def load_trajectory(traj_instance, filename):
    return traj_instance.load(filename)


def get_xyz(
    traj_instance, frames=[0, None, 1], bead_selection=None, xyz_cols=[0, 1, 2]
):
    return traj_instance.get_xyz(frames, bead_selection, xyz_cols)


def close_trajectory(traj_instance):
    traj_instance.close()


def save_pdb(chrom_dyn_obj, **kwargs):

    if chrom_dyn_obj.output_dir is None:
        chrom_dyn_obj.logger.warning("Output directory not set. Cannot save PDB.")
        return

    filename = kwargs.get(
        "filename",
        os.path.join(
            chrom_dyn_obj.output_dir,
            f"{chrom_dyn_obj.name}_{chrom_dyn_obj.simulation.currentStep}.pdb",
        ),
    )

    PBC = kwargs.get("PBC", False)

    # Unique residue names for different chains
    residue_names_by_chain = [
        "GLY",
        "ALA",
        "SER",
        "VAL",
        "THR",
        "LEU",
        "ILE",
        "ASN",
        "GLN",
        "ASP",
        "GLU",
        "PHE",
        "TYR",
        "TRP",
        "CYS",
        "MET",
        "HIS",
        "ARG",
        "LYS",
        "PRO",
    ]

    # Get atomic positions
    state = chrom_dyn_obj.simulation.context.getState(
        getPositions=True, enforcePeriodicBox=PBC
    )
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    topology = chrom_dyn_obj.topology  # OpenMM Topology

    with open(filename, "w") as pdb_file:
        pdb_file.write(f"TITLE     {chrom_dyn_obj.name}\n")
        if PBC:
            # get box vectors
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
            a, b, c = _orthorhombic_lengths(box)
            # PDB CRYST1 format: lenA lenB lenC alpha beta gamma SpaceGroup
            pdb_file.write(
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"
            )

        pdb_file.write(f"MODEL     {chrom_dyn_obj.simulation.currentStep}\n")

        atom_index = 0
        chain_index = -1
        for chain in topology.chains():
            chain_index += 1
            if chain_index > 9:
                chain_id = "9"  # Reuse chainID
            else:
                chain_id = str(chain_index)

            # Assign unique residue name per chain
            res_name = residue_names_by_chain[chain_index % len(residue_names_by_chain)]

            for residue in chain.residues():
                for atom in residue.atoms():
                    pos = positions[atom_index]
                    atom_serial = atom_index + 1
                    atom_name = "CA"  # placeholder
                    res_seq = residue.index + 1  # constant or can be residue.index + 1
                    element = "C"  # consistent with 'CA'

                    pdb_line = (
                        f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>3s} {chain_id:1s}"
                        f"{res_seq:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  "
                        f"1.00  0.00           {element:>2s}\n"
                    )
                    pdb_file.write(pdb_line)
                    atom_index += 1

        pdb_file.write("ENDMDL\n")
    chrom_dyn_obj.logger.info(f"PDB saved to {filename}")
