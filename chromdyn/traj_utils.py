#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

from __future__ import annotations
from multiprocessing import Pool, cpu_count
from typing import Dict, Union, List, Optional
from pathlib import Path
import numpy as np
import h5py
import os
import openmm.unit as unit
from openmm.app import Topology, Element
import warnings


# for GPU acceleration
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Cupy not found. GPU acceleration will not be available. Calculations will be on CPU"
    )
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

    @classmethod
    def compute_writhe_single_curve(
        cls,
        coords: np.ndarray,
        closed: bool = True,
    ) -> float:
        N = len(coords)
        if closed:
            coords = np.vstack((coords, coords[0]))  # Close the loop

        writhe = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(i - j) > 1 and not (closed and {i, j} == {0, N - 1}):
                    writhe += cls._segment_solid_angle(
                        coords[i], coords[i + 1], coords[j], coords[j + 1]
                    )
        return float(writhe)

    @classmethod
    def compute_writhe_between_curves(
        cls,
        curve1: np.ndarray,
        curve2: np.ndarray,
    ) -> float:
        """
        Computes the writhe between two closed curves (curve1 and curve2).

        Parameters
        ----------
        curve1, curve2 : ndarray of shape (N, 3) and (M, 3)
            Points defining the closed curves C1 and C2.

        Returns
        -------
        float
            The computed writhe between the two curves.
        """
        N, M = len(curve1), len(curve2)

        # Ensure curves are closed by appending the first point at the end
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
    def compute_writhe_trajectory(
        cls,
        trajectory: np.ndarray,
        closed: bool = True,
        processes: Optional[int] = None,
    ) -> List[float]:
        if processes is None:
            processes = max(cpu_count() - 1, 1)

        with Pool(processes) as pool:
            args = [(frame, closed) for frame in trajectory]
            results = pool.starmap(cls.compute_writhe_single_curve, args)

        return results

    @staticmethod
    def compute_RG(positions: np.ndarray, return_components: bool = False) -> Union[float, np.ndarray, Tuple]:
        """
        Calculates the Radius of Gyration (Rg).
        
        Args:
            positions: Coordinates array of shape (N, 3) or (T, N, 3).
            return_components: If True, returns a tuple (rg_total, rg_xyz).
                               rg_xyz will contain [rg_x, rg_y, rg_z].
        """
        # receive traj.xyz() as positions
        positions = np.asarray(positions)

        # ---------------------------------------------------------
        # Case 1: Single Frame (N, 3)
        # ---------------------------------------------------------
        if positions.ndim == 2:
            # 1. Calculate Center of Mass
            center_of_mass = np.mean(positions, axis=0)
            
            # 2. Calculate squared deviations for each dimension (x, y, z) separately
            # Shape remains (N, 3) here
            sq_deviations = (positions - center_of_mass) ** 2
            
            # 3. Mean over particles (N) to get squared Rg components
            # Shape becomes (3,) -> [Rgx^2, Rgy^2, Rgz^2]
            rg_sq_components = np.mean(sq_deviations, axis=0)
            
            # 4. Calculate Total Rg
            # Rg = sqrt(Rgx^2 + Rgy^2 + Rgz^2)
            rg_total = float(np.sqrt(np.sum(rg_sq_components)))

            if return_components:
                rg_xyz = np.sqrt(rg_sq_components) # Shape (3,)
                return rg_total, rg_xyz
            else:
                return rg_total

        # ---------------------------------------------------------
        # Case 2: Trajectory (T, N, 3)
        # ---------------------------------------------------------
        elif positions.ndim == 3:
            # 1. Calculate Centers of Mass
            # Shape (T, 3)
            centers_of_mass = np.mean(positions, axis=1)
            
            # 2. Calculate squared deviations
            # Use broadcasting: (T, N, 3) - (T, 1, 3)
            sq_deviations = (positions - centers_of_mass[:, None, :]) ** 2
            
            # 3. Mean over particles (axis 1) to get squared Rg components
            # Shape becomes (T, 3)
            rg_sq_components = np.mean(sq_deviations, axis=1)
            
            # 4. Calculate Total Rg per frame
            # Sum over xyz (axis 1 of the component array), then sqrt
            # Shape (T,)
            rg_total = np.sqrt(np.sum(rg_sq_components, axis=1))

            if return_components:
                rg_xyz = np.sqrt(rg_sq_components) # Shape (T, 3)
                return rg_total, rg_xyz
            else:
                return rg_total

        else:
            raise ValueError(
                f"positions must have shape (N, 3) or (T, N, 3), got {positions.shape}"
            )

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
        # Determine platform
        use_gpu = Analyzer._check_platform(platform)

        if use_gpu:
            return Analyzer._calculate_vacf_gpu(velocities, bead_types, sampling_step)
        else:
            return Analyzer._calculate_vacf_cpu(velocities, bead_types, sampling_step)

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

        Returns
        -------
        Dict[str, np.ndarray]
            Correlation curves and 'bin_centers'.
        """
        use_gpu = Analyzer._check_platform(platform)

        if use_gpu:
            return Analyzer._calculate_spatial_vel_corr_gpu(
                coords, velocities, bead_types, dist_range, sampling_step, num_bins
            )
        else:
            return Analyzer._calculate_spatial_vel_corr_cpu(
                coords, velocities, bead_types, dist_range, sampling_step, num_bins
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
    # Backend Implementation: VACF (GPU)
    # =========================================================================

    @staticmethod
    def _autocorrFFT_gpu(x_multi_dim: "cp.ndarray") -> "cp.ndarray":
        """(GPU) FFT-based autocorrelation helper."""
        N = x_multi_dim.shape[0]
        F = cp.fft.fft(x_multi_dim, n=2 * N, axis=0)
        res = cp.fft.ifft(F * F.conjugate(), axis=0)
        res = res[:N, ...].real

        norm_shape = [N] + [1] * (x_multi_dim.ndim - 1)
        norm = (N - cp.arange(0, N)).reshape(norm_shape)
        return res / norm

    @staticmethod
    def _calculate_vacf_gpu(velocities, bead_types, sampling_step):
        """(GPU) Implementation of VACF."""
        print("Calculating VACF on GPU...")
        type_indices = {
            utype: np.where(bead_types == utype)[0] for utype in np.unique(bead_types)
        }

        # Transfer to GPU
        sampled_vels_cp = cp.asarray(velocities[::sampling_step])

        # FFT Autocorrelation
        vacf_components_cp = Analyzer._autocorrFFT_gpu(sampled_vels_cp)

        # Sum components (x+y+z) -> (Time, Beads) -> Transpose to (Beads, Time)
        vacf_all_beads_cp = cp.sum(vacf_components_cp, axis=2).T

        results_cp = {}
        results_cp["general"] = cp.mean(vacf_all_beads_cp, axis=0)

        # Transfer indices for slicing
        for btype, indices in type_indices.items():
            if len(indices) > 0:
                indices_cp = cp.asarray(indices)
                results_cp[btype] = cp.mean(vacf_all_beads_cp[indices_cp, :], axis=0)

        # Transfer back
        return {k: cp.asnumpy(v) for k, v in results_cp.items()}

    # =========================================================================
    # Backend Implementation: VACF (CPU)
    # =========================================================================

    @staticmethod
    def _autocorrFFT_cpu(x_multi_dim: np.ndarray) -> np.ndarray:
        """(CPU) FFT-based autocorrelation helper using NumPy."""
        N = x_multi_dim.shape[0]
        # Use numpy.fft
        F = np.fft.fft(x_multi_dim, n=2 * N, axis=0)
        res = np.fft.ifft(F * F.conjugate(), axis=0)
        res = res[:N, ...].real

        norm_shape = [N] + [1] * (x_multi_dim.ndim - 1)
        norm = (N - np.arange(0, N)).reshape(norm_shape)
        return res / norm

    @staticmethod
    def _calculate_vacf_cpu(velocities, bead_types, sampling_step):
        """(CPU) Implementation of VACF using NumPy."""
        print("Calculating VACF on CPU...")
        type_indices = {
            utype: np.where(bead_types == utype)[0] for utype in np.unique(bead_types)
        }

        sampled_vels = velocities[::sampling_step]

        # FFT Autocorrelation
        vacf_components = Analyzer._autocorrFFT_cpu(sampled_vels)

        # Sum components
        vacf_all_beads = np.sum(vacf_components, axis=2).T

        results = {}
        results["general"] = np.mean(vacf_all_beads, axis=0)

        for btype, indices in type_indices.items():
            if len(indices) > 0:
                results[btype] = np.mean(vacf_all_beads[indices, :], axis=0)

        return results

    # =========================================================================
    # Backend Implementation: Spatial Corr (GPU)
    # =========================================================================
    @staticmethod
    def _calculate_spatial_vel_corr_gpu(
        coords: np.ndarray,
        velocities: np.ndarray,
        bead_types: np.ndarray,
        dist_range: float,
        sampling_step: int = 1,
        num_bins: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        (Vectorized, GPU) calculate spatial velocity correlation C(r) = <v_i · v_j>.

        In CPU, loop over frames, but calculate all O(N^2) on GPU.
        """
        print("Calculating Spatial Correlation on GPU...")
        n_frames, n_beads, _ = coords.shape

        # 1. set Bins and type pairs (lightweight on CPU)
        bins = np.linspace(0, dist_range, num_bins + 1, dtype=np.float32)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        unique_types = np.unique(bead_types)
        type_pairs = [
            "-".join(sorted(pair))
            for pair in np.array(np.meshgrid(unique_types, unique_types)).T.reshape(
                -1, 2
            )
        ]
        type_pairs = sorted(list(set(type_pairs)))

        # --- 2. RATIONALE: pre-calculate masks and *once* transfer to GPU ---
        rows, cols = np.triu_indices(n_beads, k=1)
        rows_cp = cp.asarray(rows)
        cols_cp = cp.asarray(cols)

        bead_types_i = bead_types[rows]
        bead_types_j = bead_types[cols]

        pair_masks_cp = {}
        for key in type_pairs:
            t1, t2 = key.split("-")
            if t1 == t2:
                mask = (bead_types_i == t1) & (bead_types_j == t2)
            else:
                mask = ((bead_types_i == t1) & (bead_types_j == t2)) | (
                    (bead_types_i == t2) & (bead_types_j == t1)
                )
            pair_masks_cp[key] = cp.asarray(mask)

        bins_cp = cp.asarray(bins)

        # 3. RATIONALE: initialize accumulators on GPU
        total_corr_cp = {key: cp.zeros(num_bins) for key in ["general"] + type_pairs}
        counts_cp = {key: cp.zeros(num_bins) for key in ["general"] + type_pairs}

        # 4. loop over sampled frames on CPU
        for frame_idx in range(0, n_frames, sampling_step):

            # 5. RATIONALE: transfer *only the current frame* to GPU
            frame_coords_cp = cp.asarray(coords[frame_idx], dtype=cp.float32)
            frame_vels_cp = cp.asarray(velocities[frame_idx], dtype=cp.float32)

            # 6. RATIONALE: execute all O(N^2) calculations on GPU

            # a. distance matrix: use broadcast (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> (N, N)
            dist_matrix_cp = cp.linalg.norm(
                frame_coords_cp[:, None, :] - frame_coords_cp[None, :, :], axis=2
            )

            # b. dot product matrix: (N, 3) @ (3, N) -> (N, N)
            v_dot_v_cp = frame_vels_cp @ frame_vels_cp.T

            # c. extract upper triangle
            all_dists_cp = dist_matrix_cp[rows_cp, cols_cp]
            all_dots_cp = v_dot_v_cp[rows_cp, cols_cp]

            # d. digitize
            all_bin_indices_cp = cp.digitize(all_dists_cp, bins_cp[1:])

            # 7. RATIONALE: use bincount on GPU for vectorized accumulation
            valid_mask_cp = all_bin_indices_cp < num_bins

            # accumulate 'general'
            valid_bins_cp = all_bin_indices_cp[valid_mask_cp]
            valid_dots_cp = all_dots_cp[valid_mask_cp]
            total_corr_cp["general"] += cp.bincount(
                valid_bins_cp, weights=valid_dots_cp, minlength=num_bins
            )
            counts_cp["general"] += cp.bincount(valid_bins_cp, minlength=num_bins)

            # accumulate by type
            for key, type_mask_cp in pair_masks_cp.items():
                final_mask_cp = valid_mask_cp & type_mask_cp
                if cp.any(final_mask_cp):
                    type_bins_cp = all_bin_indices_cp[final_mask_cp]
                    type_dots_cp = all_dots_cp[final_mask_cp]
                    total_corr_cp[key] += cp.bincount(
                        type_bins_cp, weights=type_dots_cp, minlength=num_bins
                    )
                    counts_cp[key] += cp.bincount(type_bins_cp, minlength=num_bins)

        # 8. calculate final average on GPU
        results_cp = {}
        for key in total_corr_cp:

            # replace where to be compatible with old version of cupy

            # 1. copy counts to avoid modifying original accumulators
            counts_safe_cp = counts_cp[key].copy()

            # 2. create mask for all bins with count 0
            zero_mask_cp = counts_safe_cp == 0

            # 3. replace these 0s with 1.0. This doesn't affect the result,
            #    because we will set these positions to NaN later.
            counts_safe_cp[zero_mask_cp] = 1.0

            # 4. perform regular division (now safe)
            corr_cp = total_corr_cp[key] / counts_safe_cp

            # 5. set positions marked as 0 to NaN
            corr_cp[zero_mask_cp] = cp.nan

            results_cp[key] = corr_cp

        # 9. transfer final small result array back to CPU
        results_np = {k: cp.asnumpy(v) for k, v in results_cp.items()}
        results_np["bin_centers"] = bin_centers

        return results_np

    # =========================================================================
    # Backend Implementation: Spatial Corr (CPU)
    # =========================================================================

    @staticmethod
    def _calculate_spatial_vel_corr_cpu(
        coords, velocities, bead_types, dist_range, sampling_step, num_bins
    ):
        """(CPU) Vectorized spatial velocity correlation using NumPy."""
        print("Calculating Spatial Velocity Correlation on CPU...")

        n_frames, n_beads, _ = coords.shape
        bins = np.linspace(0, dist_range, num_bins + 1, dtype=np.float32)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        unique_types = np.unique(bead_types)
        type_pairs = sorted(
            list(
                set(
                    [
                        "-".join(sorted(pair))
                        for pair in np.array(
                            np.meshgrid(unique_types, unique_types)
                        ).T.reshape(-1, 2)
                    ]
                )
            )
        )

        # Pre-calc indices (CPU is efficient with indexing)
        rows, cols = np.triu_indices(n_beads, k=1)

        bead_types_i = bead_types[rows]
        bead_types_j = bead_types[cols]

        pair_masks = {}
        for key in type_pairs:
            t1, t2 = key.split("-")
            if t1 == t2:
                mask = (bead_types_i == t1) & (bead_types_j == t2)
            else:
                mask = ((bead_types_i == t1) & (bead_types_j == t2)) | (
                    (bead_types_i == t2) & (bead_types_j == t1)
                )
            pair_masks[key] = mask

        total_corr = {key: np.zeros(num_bins) for key in ["general"] + type_pairs}
        counts = {key: np.zeros(num_bins) for key in ["general"] + type_pairs}

        # Loop over frames (Vectorized inside frame)
        for frame_idx in range(0, n_frames, sampling_step):
            frame_coords = coords[frame_idx]  # (N, 3)
            frame_vels = velocities[frame_idx]  # (N, 3)

            # a. Distance Matrix (Broadcasting)
            # Warning: For very large N (>5000), this creates a large N*N matrix.
            # CPU RAM is usually sufficient, but be aware.
            dist_matrix = np.linalg.norm(
                frame_coords[:, None, :] - frame_coords[None, :, :], axis=2
            )

            # b. Dot Product
            v_dot_v = frame_vels @ frame_vels.T

            # c. Extract upper triangle
            all_dists = dist_matrix[rows, cols]
            all_dots = v_dot_v[rows, cols]

            # d. Digitize
            all_bin_indices = np.digitize(all_dists, bins[1:])

            # e. Accumulate (using np.bincount)
            valid_mask = all_bin_indices < num_bins

            # General
            valid_bins = all_bin_indices[valid_mask]
            valid_dots = all_dots[valid_mask]

            if valid_bins.size > 0:
                total_corr["general"] += np.bincount(
                    valid_bins, weights=valid_dots, minlength=num_bins
                )
                counts["general"] += np.bincount(valid_bins, minlength=num_bins)

            # By Type
            for key, type_mask in pair_masks.items():
                final_mask = valid_mask & type_mask
                if np.any(final_mask):
                    type_bins = all_bin_indices[final_mask]
                    type_dots = all_dots[final_mask]
                    total_corr[key] += np.bincount(
                        type_bins, weights=type_dots, minlength=num_bins
                    )
                    counts[key] += np.bincount(type_bins, minlength=num_bins)

        # Final Average
        results = {}
        for key in total_corr:
            counts_safe = counts[key].copy()
            # Safe division for CPU (avoiding runtime warnings)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = total_corr[key] / counts_safe

            # Set 0 counts to NaN
            corr[counts_safe == 0] = np.nan
            results[key] = corr

        results["bin_centers"] = bin_centers
        return results


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
        pos: list[np.ndarray] = []
        with h5py.File(str(traj_file), "r") as f:
            for key in sorted(f.keys()):
                try:
                    frame_id = int(key)
                    if frame_id % d == 0:
                        pos.append(np.array(f[key]))
                except ValueError:
                    # Ignore keys that are not integer frame IDs
                    pass

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

        self.cndb = h5py.File(filename, "r")

        # Sort frame keys
        frame_keys = sorted([k for k in self.cndb.keys() if k.isdigit()], key=int)
        self.n_frames = len(frame_keys)

        if self.n_frames == 0:
            print("Warning: No frames found in file.")
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
            print("  Warning: 'types' dataset not found. Assuming uniform bead types.")
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
                print(f"  Warning: Failed to load topology data from HDF5: {e}")

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

        print(f"Loaded {self.filename}: {self.n_frames} frames, {self.n_beads} beads.")
        if self.topology:
            print(
                f"Topology loaded: {self.topology.getNumAtoms()} atoms, {self.topology.getNumBonds()} bonds"
            )
        if self.box_vectors is not None:
            print(f"Box vectors loaded. Shape: {self.box_vectors.shape}")

        return self

    def xyz(self, frames=[0, None, 1], bead_selection=None, xyz_cols=[0, 1, 2]):
        """
        Get the selected beads' 3D position from a **cndb** file for multiple frames.
        """
        if self.cndb is None:
            raise RuntimeError("No file loaded. Call load() first.")

        frame_list = []

        if bead_selection is None:
            selection = np.arange(self.n_beads)
        else:
            selection = np.array(bead_selection)

        start, end, step = frames
        if end is None:
            end = self.n_frames

        # Range check
        start = max(0, start)
        end = min(end, self.n_frames)

        for i in range(start, end, step):
            try:
                key = str(i)
                if key not in self.cndb:
                    continue
                frame_data = np.array(self.cndb[key])
                selected_data = np.take(
                    np.take(frame_data, selection, axis=0), xyz_cols, axis=1
                )
                frame_list.append(selected_data)
            except KeyError:
                print(f"Warning: Frame {i} doesn't exist, skipping.")
            except Exception as e:
                print(f"Error extracting data from frame {i}: {e}")

        return np.array(frame_list)

    def close(self):
        """Close the HDF5 file handle."""
        if hasattr(self, "cndb") and self.cndb:
            self.cndb.close()

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
    def compute_rg_type(self, get_components: bool = False):
        """
        Function to compute Radius of Gyration (Rg) classified by particle type.

        Parameters:
            get_components (bool): If True, returns both total Rg and its XYZ components.
                                Default is False.

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
        bead_types = np.asarray(self.chrom_seq)

        # 2. Initialize result dictionary
        results = {}

        # 3. Calculate 'general' Rg (all beads)
        # Forward the get_components flag to the static method
        results["general"] = Analyzer.compute_RG(all_positions, return_components=get_components)

        # 4. Check if system is Heterogeneous (异质系统)
        unique_types = np.unique(bead_types)

        # If type count is greater than 1, calculate by type
        if len(unique_types) > 1:
            for t_type in unique_types:
                # Create Boolean Mask (布尔掩码)
                mask = (bead_types == t_type)

                # Slice positions: [all frames, filtered beads, xyz]
                subset_positions = all_positions[:, mask, :]

                # Ensure type name is string format as key
                key_name = str(t_type)

                # 5. Calculate Rg for this type
                # The return type of Analyzer.compute_RG depends on get_components
                rg_data = Analyzer.compute_RG(subset_positions, return_components=get_components)
                
                # If get_components is True, rg_data is a tuple (total, xyz)
                # We can store it as a sub-dictionary for better readability
                if get_components:
                    total_rg, xyz_rg = rg_data
                    results[key_name] = {
                        "total": total_rg,
                        "components": xyz_rg
                    }
                    # Also update "general" to a dict format for structure consistency
                    if key_name == str(unique_types[0]): # Only need to format "general" once
                        gen_total, gen_xyz = results["general"]
                        results["general"] = {"total": gen_total, "components": gen_xyz}
                else:
                    results[key_name] = rg_data

        return results


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
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    topology = chrom_dyn_obj.topology  # OpenMM Topology

    with open(filename, "w") as pdb_file:
        pdb_file.write(f"TITLE     {chrom_dyn_obj.name}\n")
        if PBC:
            # get box vectors
            box = chrom_dyn_obj.simulation.context.getState().getPeriodicBoxVectors()
            a = box[0].x * 10.0  # nm to Angstrom for PDB
            b = box[1].y * 10.0
            c = box[2].z * 10.0
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
