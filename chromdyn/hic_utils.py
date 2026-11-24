#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

import numpy as np
import os
from .utilities import LogManager
from .traj_utils import _sorted_frame_keys, _orthorhombic_lengths
from scipy.ndimage import median_filter, uniform_filter

# from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
import h5py
from typing import Optional

# from .traj_utils import Trajectory

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class HiCManager:
    def __init__(
        self,
        logger=None,
    ):
        """
        HiCManager to handle Hi-C contact maps from file, array, or generate random.

        Inputs are passed to get_numpy_from_hic. Random maps have values in
        [0, 1), are symmetric, and exclude self contacts on the diagonal.
        """
        self.logger = logger or LogManager().get_logger(__name__)
        self.hic_map = None

    def _remove_neighbor(self, hicmap, k):
        offsets = [k] if isinstance(k, (int, np.integer)) else k
        if not isinstance(offsets, list) or any(not isinstance(i, (int, np.integer)) for i in offsets):
            raise ValueError("k must be an integer or a list of integers")
        res = np.array(hicmap, copy=True)
        for offset in offsets:
            offset = abs(offset)
            if offset >= len(res):
                continue
            rows = np.arange(len(res)-offset)
            res[rows, rows+offset] = 0
            res[rows+offset, rows] = 0
        return np.triu(res) + np.triu(res, k=1).T

    def filter_matrix(self, matrix, window_size=3, method="median", padding="wrap"):
        """
        Apply a sliding window filter (median or mean) to a matrix, keeping the same dimensions.

        Args:
            matrix (np.ndarray): Input 2D matrix (array).
            window_size (int or tuple): Size of the filtering window (e.g., 3 or (3, 3)).
            method (str): 'median' or 'mean' (average).
            padding (str): Padding mode for boundaries ('reflect', 'constant', 'nearest', 'mirror', 'wrap').

        Returns:
            np.ndarray: Filtered matrix with same dimensions.
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        if method == "median":
            result = median_filter(matrix, size=window_size, mode=padding)
        elif method == "mean" or method == "average":
            result = uniform_filter(matrix, size=window_size, mode=padding)
        else:
            raise ValueError("Unsupported method. Choose 'median' or 'mean'.")

        return result

    def symmetric_l1_normalization(self, A):
        """
        Perform symmetric L1 normalization on a square symmetric matrix.

        Args:
            A (numpy.ndarray): Input symmetric matrix (NxN).

        Returns:
            numpy.ndarray: Symmetrically normalized matrix.
        """
        if not np.allclose(A, A.T):
            raise ValueError("Input matrix must be symmetric.")

        # Compute absolute row and column sums (same since A is symmetric)
        row_sums = np.nansum(np.abs(A), axis=1)  # Sum along rows
        col_sums = np.nansum(np.abs(A), axis=0)  # Sum along columns

        # Compute the normalization factor using sqrt of row and column sums product
        normalization_factors = np.sqrt(np.outer(row_sums, col_sums))

        # Normalize each element
        A_normalized = A / normalization_factors

        return A_normalized

    def get_flat_matrix(self, mat):
        """
        Get the averaged Hi-C map.
        Normalize pairwise contact frequencies by the average frequency between similarly distant genomic loci.

        Args:
            mat (np.ndarray): Input Hi-C contact matrix.
            tol (float): Tolerance to avoid division by zero, default is 1e-8.

        Returns:
            np.ndarray: Averaged Hi-C contact matrix.
        """
        avg_mat = np.zeros_like(mat)
        n = mat.shape[0]

        # Loop through each diagonal of the matrix
        for i in range(n):
            # Normalize diagonal values by the average frequency and add to avg_mat
            avg_val = np.nanmean(np.diag(mat, k=i))
            idx = np.arange(n - i)
            # Set the upper diagonal
            avg_mat[idx, idx + i] = avg_val
            # Set the lower diagonal if not the main diagonal
            if i > 0:
                avg_mat[idx + i, idx] = avg_val

        return avg_mat

    def update_kth_neighbor(self, matrix, k, val):
        res = matrix.copy()
        n = matrix.shape[0]
        idx = np.arange(n - k)
        res[idx, idx + k] = val
        res[idx + k, idx] = val
        return res

    def normalize_by_kth_neighbors(self, matrix, k):
        res = matrix / np.nanmean(np.diag(matrix, k=k))
        return res

    def normalize_hic(self, matrix, p2=0.5):
        res = self.filter_matrix(matrix, window_size=1)
        res = self._remove_neighbor(res, k=[0, 1])
        res[0, -1] = 0.0
        res[-1, 0] = 0.0
        self.logger.debug("Symmetric: %s", self.check_symmetric(res))
        res = self.symmetric_l1_normalization(res)
        self.logger.debug("Symmetric: %s", self.check_symmetric(res))
        # res = self.update_kth_neighbor(res, k=1, val=res.max())
        second_neighbor_counts = np.mean(np.diag(res, k=2))
        res *= p2 / second_neighbor_counts
        res = self.update_kth_neighbor(res, k=1, val=1.0)
        res[0, -1] = 1.0
        res[-1, 0] = 1.0
        np.fill_diagonal(res, 0.0)
        # res = self.normalize_by_kth_neighbors(res, k=1)
        return res

    def get_numpy_from_hic(self, hicmap, skiprows=0, skipcols=0):
        """
        Attempts to load or generate Hi-C map based on the type of input.

        Args:
            hicmap (str, np.ndarray, int): Hi-C data source.
        """

        # First, try to interpret as a file path if it's a string
        if isinstance(hicmap, str):
            try:
                if os.path.isfile(hicmap):
                    if hicmap.endswith(".txt"):
                        hic = np.loadtxt(hicmap, skiprows=skiprows, dtype=str)
                        if skipcols == 1:
                            hic = np.delete(hic, 0, axis=1)
                        hic = np.array(hic, dtype=float)
                        self.hic_map = hic
                        self.logger.info(
                            f"Hi-C map loaded from file '{hicmap}' with shape {hic.shape}."
                        )
                        return hic  # Success, exit function
                    else:
                        self.logger.warning(
                            f"File '{hicmap}' found but unsupported format. Expecting .txt."
                        )
                else:
                    self.logger.warning(f"Path '{hicmap}' is not a valid file.")
            except Exception as e:
                self.logger.error(f"Failed to load Hi-C map from file '{hicmap}': {e}")

        # If not a valid file or failed, check if it's a NumPy array
        if isinstance(hicmap, np.ndarray):
            try:
                if hicmap.ndim != 2:
                    raise ValueError(
                        f"Provided NumPy array must be 2D. Got shape: {hicmap.shape}"
                    )
                hic = hicmap
                self.hic_map = hic
                self.logger.info(
                    f"Hi-C map loaded from NumPy array with shape {hicmap.shape}."
                )
                return hic  # Success, exit function
            except Exception as e:
                self.logger.error(f"Failed to use provided NumPy array: {e}")

        # If not an array, check if it's an integer for random generation
        if isinstance(hicmap, int):
            try:
                if hicmap <= 0:
                    raise ValueError(
                        "Integer for Hi-C map generation must be positive."
                    )
                random_map = np.random.random(size=(hicmap, hicmap))
                hic = (random_map + random_map.T) / 2
                np.fill_diagonal(hic, 0)
                self.hic_map = hic
                self.logger.info(
                    f"Random symmetric Hi-C map generated (size: {hicmap}x{hicmap}, range: [0, 1))."
                )
                return hic  # Success, exit function
            except Exception as e:
                self.logger.error(f"Failed to generate random Hi-C map: {e}")

        # If none of the above succeeded, raise a clear error
        raise TypeError(
            f"[FATAL ERROR] Unable to interpret input '{hicmap}'. Must be path to '.txt' file, 2D NumPy array, or positive integer."
        )

    def get_hic_map(self):
        """Returns the loaded Hi-C map."""
        return self.hic_map

    def get_shape(self):
        """Returns shape of the Hi-C map."""
        return self.hic_map.shape if self.hic_map is not None else None

    def get_Ps(self, hic):
        Ps = []
        Ps_std = []
        for ii in range(hic.shape[0]):
            Ps.append(np.nanmean(np.diag(hic, k=ii)))
            Ps_std.append(np.nanstd(np.diag(hic, k=ii)))
        # ax.errorbar(list(range(len(Ps))), Ps, yerr=Ps_std, fmt='.')
        return Ps, Ps_std

    def condense_matrix(self, matrix):
        N = matrix.shape[0]  # Original matrix size
        if N % 2 != 0:
            # raise ValueError("Matrix size must be divisible by 2")
            matrix = matrix[:-1, :-1]
            N = matrix.shape[0]

        n = N // 2  # New matrix size
        # Reshape the matrix into blocks of 2x2
        reshaped = matrix.reshape(n, 2, n, 2).swapaxes(1, 2)
        # Average over the last two axes to condense the blocks
        condensed_matrix = reshaped.mean(axis=(2, 3))

        return condensed_matrix

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def cndb_to_numpy(self, traj_file, skip_frames=1):
        self.logger.info("Loading trajectory ...")
        with h5py.File(traj_file, "r") as pos:
            keys = self._selected_frame_keys(pos, skip_frames)
            xyz = np.array([pos[key][...] for key in keys])
        self.logger.info(f"Trajectory shape: {xyz.shape}")
        return xyz

    @staticmethod
    def _selected_frame_keys(handle, stride):
        if not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError("skip_frames must be a positive integer")
        keys = _sorted_frame_keys(handle)[::stride]
        if not keys:
            raise ValueError("No trajectory frames found in CNDB file")
        return keys

    @staticmethod
    def _divide_into_subtraj(xyz, num_proc):
        return [chunk for chunk in np.array_split(xyz, num_proc) if len(chunk)]

    def _resolve_platform(self, platform):
        platform = platform.upper()
        if platform not in {"CPU", "CUDA"}:
            raise ValueError("platform must be 'CPU' or 'CUDA'")
        if platform == "CUDA":
            available = False
            if CUPY_AVAILABLE:
                try:
                    available = cp.cuda.runtime.getDeviceCount() > 0
                except cp.cuda.runtime.CUDARuntimeError as exc:
                    self.logger.warning(f"CUDA runtime unavailable: {exc}")
            if not available:
                self.logger.warning("CUDA unavailable. Falling back to CPU.")
                return "CPU"
        return platform

    def gen_hic_from_cndb(
        self,
        traj_file,
        mu = 2.0,
        rc = 2.0,
        p=None,
        platform="CPU",
        parallel: bool = True,  # RE-INTRODUCED this parameter
        skip_frames=1,
        batch_size: Optional[int] = None,
    ):
        """
        (Version 3.1: Corrected handling of the 'parallel' parameter for CPU mode)
        Generates a Hi-C matrix from a CNDB trajectory file.

        Args:
            ... (your existing args)
            platform (str): The computation platform. 'CPU' (default) or 'CUDA'.
            parallel (bool): If platform is 'CPU', this flag determines whether to use
                            multiprocessing (True) or run serially (False).
                            Defaults to True.
            skip_frames (int): ...
            batch_size (int, optional): If platform is 'CUDA', specifies the number of frames
                                        to process in each GPU batch for memory efficiency.
                                        If None, processes all frames at once.
        """
        platform = self._resolve_platform(platform)
        if platform.upper() == "CPU" and batch_size is not None:
            self.logger.warning(
                "`batch_size` is specified but `platform` is 'CPU'. The batching parameter will be ignored."
            )

        if platform.upper() == "CPU":
            if parallel:
                self.logger.warning(
                    "CPU HiC now streams CNDB frames to avoid OOM; multiprocessing is disabled for this path."
                )

            self.logger.info("Computing HiC serially on CPU with streaming CNDB reads...")
            hic = self._gen_hic_from_cndb_streaming(
                traj_file=traj_file,
                mu=mu,
                rc=rc,
                p=p,
                skip_frames=skip_frames,
            )
            self.logger.info(f"Generated HiC matrix of shape: {hic.shape}")
            return hic

        if platform.upper() == "CUDA":
            self.logger.info("Computing HiC on CUDA platform with streaming CNDB reads...")
            hic = self._gen_hic_from_cndb_gpu_streaming(
                traj_file=traj_file,
                mu=mu,
                rc=rc,
                p=p,
                skip_frames=skip_frames,
                batch_size=batch_size,
            )
            self.logger.info(f"Generated HiC matrix of shape: {hic.shape}")
            return hic

    def _gen_hic_from_cndb_streaming(
        self,
        traj_file,
        mu=2.0,
        rc=2.0,
        p=None,
        skip_frames=1,
        progress_interval=50000,
    ):
        self.logger.info("Streaming trajectory from CNDB ...")
        with h5py.File(traj_file, "r") as pos:
            frame_keys = self._selected_frame_keys(pos, skip_frames)
            first_frame = np.asarray(pos[frame_keys[0]], dtype=np.float32)
            n_beads = first_frame.shape[0]
            hic = np.zeros((n_beads, n_beads), dtype=np.float64)

            for frame_count, frame_key in enumerate(frame_keys, start=1):
                frame = np.asarray(pos[frame_key], dtype=np.float32)
                hic += _calc_prob(frame, mu, rc, p)
                if progress_interval and frame_count % progress_interval == 0:
                    self.logger.info(
                        f"Streaming HiC progress: {frame_count}/{len(frame_keys)} frames"
                    )

        hic /= len(frame_keys)
        self.logger.info(f"Trajectory frames processed: {len(frame_keys)}")
        return hic

    def _gen_hic_from_cndb_gpu_streaming(
        self,
        traj_file,
        mu=2.0,
        rc=2.0,
        p=None,
        skip_frames=1,
        batch_size: Optional[int] = None,
    ):
        """Stream CNDB frames in batches to GPU without loading full trajectory."""
        if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        with h5py.File(traj_file, "r") as pos:
            frame_keys = self._selected_frame_keys(pos, skip_frames)
            n_frames = len(frame_keys)
            first_frame = np.asarray(pos[frame_keys[0]], dtype=np.float32)
            n_beads = first_frame.shape[0]
            chunk_size = min(n_frames, 256) if batch_size is None else int(batch_size)
            cumulative_prob = cp.zeros((n_beads, n_beads), dtype=cp.float64)
            diag_idx = cp.arange(n_beads)
            try:
                for start in range(0, n_frames, chunk_size):
                    batch_keys = frame_keys[start : start + chunk_size]
                    batch_np = np.empty((len(batch_keys), n_beads, 3), dtype=np.float32)
                    for idx, frame_key in enumerate(batch_keys):
                        batch_np[idx] = np.asarray(pos[frame_key], dtype=np.float32)
                    _accumulate_hic_gpu_batch(
                        cumulative_prob, batch_np, mu, rc, p, None, diag_idx
                    )
                    del batch_np
                hic_matrix = cp.asnumpy(cumulative_prob / n_frames)
            finally:
                del cumulative_prob, diag_idx
                _free_cupy_memory_pool()
        return hic_matrix

    # =========================================================================
    # PBC Hi-C Generation (Fixed: GPU Summation & Precision)
    # =========================================================================
    def gen_pbc_hic_from_cndb(
        self,
        traj_file,
        mu=2.0,
        rc=2.0,
        p=None,
        platform="CPU",
        parallel=True,
        skip_frames=1,
        batch_size=None,
    ):
        """
        Generates a Hi-C matrix from a CNDB trajectory file using PBC (Minimum Image Convention)
        and a smooth probability function (Sigmoid/Power-law).

        It uses the standard Trajectory class to load data and box vectors.

        Args:
            traj_file (str): Path to .cndb file.
            mu (float): Steepness parameter for the sigmoid function.
            rc (float): Cutoff distance (nm) for the sigmoid transition.
            p (float, optional): Power-law exponent for r > rc. If None, uses sigmoid everywhere.
            platform (str): 'CPU' or 'CUDA'.
            parallel (bool): Use multiprocessing for CPU.
            skip_frames (int): Stride for reading frames.
            batch_size (int): Batch size for GPU calculation.
        """
        from .traj_utils import Trajectory

        platform = self._resolve_platform(platform)
        self.logger.info(f"Computing PBC Hi-C with mu={mu}, rc={rc}, p={p}...")
        # --- CUDA Platform ---
        if platform == "CUDA":
            hic = self._gen_pbc_hic_from_cndb_gpu_streaming(
                traj_file=traj_file,
                mu=mu,
                rc=rc,
                p=p,
                skip_frames=skip_frames,
                batch_size=batch_size,
            )
            self.logger.info(f"Generated PBC HiC matrix of shape: {hic.shape}")
            return hic

        traj = Trajectory(traj_file)
        try:
            xyz = traj.xyz(frames=[0, None, skip_frames])
            if not len(xyz):
                raise ValueError("No trajectory frames selected")
            if traj.box_vectors is not None:
                boxes = traj.box_vectors[::skip_frames].copy()
                norms = np.linalg.norm(boxes, axis=(1, 2))
                if np.any(norms < 1e-6):
                    self.logger.warning("Zero box vectors: disabling PBC for those frames.")
                    boxes[norms < 1e-6] = np.eye(3)*999999.9
            else:
                self.logger.warning("No box vectors. Assuming infinite box.")
                boxes = np.tile(np.eye(3)*999999.9, (len(xyz), 1, 1))
            _orthorhombic_lengths(boxes)
        finally:
            traj.close()

        self.logger.info(
            f"Trajectory shape: {xyz.shape}, Box data shape: {boxes.shape}"
        )

        # --- CPU Platform ---
        if parallel:
            try:
                import multiprocessing

                num_proc = min(
                    multiprocessing.cpu_count(), 32, max(1, xyz.shape[0] // 10)
                )

                # Split both coordinates and boxes for multiprocessing
                sub_xyz_list = self._divide_into_subtraj(xyz, num_proc)
                sub_box_list = self._divide_into_subtraj(boxes, num_proc)

                self.logger.info(
                    f"Using multiprocessing on CPU ({num_proc} processes)."
                )

                # Pass mu, rc, p to workers
                args_list = [
                    (sub_xyz_list[i], sub_box_list[i], mu, rc, p)
                    for i in range(len(sub_xyz_list))
                ]

                hic = np.zeros((xyz.shape[1], xyz.shape[1]), dtype=np.float32)

                with multiprocessing.get_context("spawn").Pool(processes=num_proc) as pool:
                    results = pool.map(_wrap_calc_pbc, args_list)
                    total_frames = xyz.shape[0]
                    for i, res in enumerate(results):
                        weight = sub_xyz_list[i].shape[0] / total_frames
                        hic += res * weight

            except Exception as e:
                self.logger.warning(
                    f"Multiprocessing failed: {e}. Falling back to serial."
                )
                parallel = False

        if not parallel:
            self.logger.info("Computing PBC HiC serially on CPU...")
            hic = _calc_pbc_hic_cpu_serial(xyz, boxes, mu, rc, p)

        self.logger.info(f"Generated PBC HiC matrix of shape: {hic.shape}")
        return hic

    def _gen_pbc_hic_from_cndb_gpu_streaming(
        self,
        traj_file,
        mu=2.0,
        rc=2.0,
        p=None,
        skip_frames=1,
        batch_size: Optional[int] = None,
    ):
        """Stream CNDB frames and boxes in batches to GPU for PBC Hi-C."""
        from .traj_utils import Trajectory

        if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")

        traj = Trajectory(traj_file)
        try:
            frame_positions = list(traj._frame_positions([0, None, skip_frames]))
            if not frame_positions:
                raise ValueError("No trajectory frames selected")
            n_frames = len(frame_positions)
            n_beads = traj.n_beads
            if traj.box_vectors is not None:
                boxes = traj.box_vectors[frame_positions].copy()
                norms = np.linalg.norm(boxes, axis=(1, 2))
                if np.any(norms < 1e-6):
                    self.logger.warning("Zero box vectors: disabling PBC for those frames.")
                    boxes[norms < 1e-6] = np.eye(3) * 999999.9
            else:
                self.logger.warning("No box vectors. Assuming infinite box.")
                boxes = np.tile(np.eye(3) * 999999.9, (n_frames, 1, 1))
            boxes_diag = _orthorhombic_lengths(boxes)
            use_pbc = bool(np.min(boxes_diag) < 1e5)

            self.logger.info(
                f"Trajectory shape: ({n_frames}, {n_beads}, 3), Box data shape: {boxes.shape}"
            )
            chunk_size = min(n_frames, 256) if batch_size is None else int(batch_size)
            cumulative_prob = cp.zeros((n_beads, n_beads), dtype=cp.float64)
            diag_idx = cp.arange(n_beads)
            try:
                for start in range(0, n_frames, chunk_size):
                    end = min(start + chunk_size, n_frames)
                    batch_pos = np.empty((end - start, n_beads, 3), dtype=np.float32)
                    for idx, frame_idx in enumerate(frame_positions[start:end]):
                        batch_pos[idx] = np.asarray(
                            traj.cndb[traj.frame_keys[frame_idx]], dtype=np.float32
                        )
                    batch_box = (
                        boxes_diag[start:end].astype(np.float32, copy=False)
                        if use_pbc
                        else None
                    )
                    _accumulate_hic_gpu_batch(
                        cumulative_prob, batch_pos, mu, rc, p, batch_box, diag_idx
                    )
                    del batch_pos
                return cp.asnumpy(cumulative_prob / n_frames)
            finally:
                del cumulative_prob, diag_idx
                _free_cupy_memory_pool()
        finally:
            traj.close()


def _calc_prob(data, mu, rc, p=None):
    """
    Calculates a Hi-C like contact probability matrix from 3D coordinate data.

    This modified version supports two modes:
    1. If 'p' is provided, it uses a hybrid model: a sigmoid function for distances
    less than or equal to rc, and a power-law decay for distances greater than rc.
    2. If 'p' is None, it uses only the sigmoid function for all distances.

    Args:
        data (np.ndarray): The input coordinate data, shape (N, 3).
        mu (float): Steepness parameter for the sigmoid function.
        rc (float): Cutoff distance for the sigmoid function.
        p (float, optional): The exponent for the power-law decay. If None,
                            only the sigmoid function is used. Defaults to None.

    Returns:
        np.ndarray: An N x N symmetric matrix of contact probabilities.
    """
    # Compute condensed distance matrix (upper triangle of the distance matrix)
    r = pdist(data, metric="euclidean")

    # Check if the power-law exponent 'p' is provided
    if p is not None:
        # If p is provided, use the original conditional logic
        # applying sigmoid for r <= rc and power-law for r > rc.
        f_condensed = np.where(
            r <= rc, 0.5 * (1 + np.tanh(mu * (rc - r))), 0.5 * (rc / r) ** p
        )
    else:
        # If p is None, apply only the sigmoid-like function to all distances.
        f_condensed = 0.5 * (1 + np.tanh(mu * (rc - r)))

    # Convert the condensed (1D) array back to a square symmetric matrix
    f = squareform(f_condensed)
    return f


def _free_cupy_memory_pool() -> None:
    if CUPY_AVAILABLE and hasattr(cp, "get_default_memory_pool"):
        cp.get_default_memory_pool().free_all_blocks()


def _accumulate_hic_gpu_batch(
    cumulative_prob,
    batch_pos_np,
    mu,
    rc,
    p,
    batch_boxes_diag_np,
    diag_idx,
) -> None:
    batch_pos = cp.asarray(batch_pos_np, dtype=cp.float32)
    batch_box = (
        cp.asarray(batch_boxes_diag_np, dtype=cp.float32)
        if batch_boxes_diag_np is not None
        else None
    )
    try:
        for j in range(batch_pos.shape[0]):
            pos = batch_pos[j]
            diff = pos[:, cp.newaxis, :] - pos[cp.newaxis, :, :]
            if batch_box is not None:
                box = batch_box[j]
                diff -= box * cp.round(diff / box)
            r = cp.linalg.norm(diff, axis=-1)
            if p is not None:
                r = cp.where(r == 0, 1e-10, r)
                term_power = 0.5 * (rc / r) ** p
                prob = cp.where(r <= rc, 0.5 * (1 + cp.tanh(mu * (rc - r))), term_power)
            else:
                prob = 0.5 * (1 + cp.tanh(mu * (rc - r)))
            prob[diag_idx, diag_idx] = 0.0
            cumulative_prob += prob
    finally:
        del batch_pos, batch_box


def _calc_HiC_from_traj_array_gpu(traj, mu, rc, p=None, boxes=None, batch_size=None):
    """
    Fully vectorized Hi-C calculation on GPU using CuPy.
    Transfers only ``batch_size`` frames at a time to GPU and accumulates in float64.
    Handles both PBC (if boxes provided) and Non-PBC (if boxes is None).
    """
    n_frames, n_beads, _ = traj.shape
    if n_frames == 0:
        raise ValueError("Trajectory must contain at least one frame")
    if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")

    # --- 1. PBC Setup ---
    use_pbc = False
    boxes_diag = None
    if boxes is not None:
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        boxes_diag = np.diagonal(boxes_arr, axis1=1, axis2=2)
        if np.min(boxes_diag) < 1e5:
            use_pbc = True

    chunk_size = min(n_frames, 256) if batch_size is None else int(batch_size)
    cumulative_prob = cp.zeros((n_beads, n_beads), dtype=cp.float64)
    diag_idx = cp.arange(n_beads)

    # --- 2. Computation Loop (stream batches from host to GPU) ---
    try:
        for i in range(0, n_frames, chunk_size):
            end = min(i + chunk_size, n_frames)
            batch_box = boxes_diag[i:end] if use_pbc else None
            _accumulate_hic_gpu_batch(
                cumulative_prob, traj[i:end], mu, rc, p, batch_box, diag_idx
            )
        hic_matrix = cp.asnumpy(cumulative_prob / n_frames)
    finally:
        del cumulative_prob, diag_idx
        _free_cupy_memory_pool()

    return hic_matrix


# Function to wrap single-call processing (needed for Pool.map)
def _wrap_calc(subtraj_mu_rc_p):
    subtraj, mu, rc, p = subtraj_mu_rc_p
    return _calc_HiC_from_traj_array(subtraj, mu, rc, p).astype(np.float32)


def _wrap_calc_pbc(args):
    sub_xyz, sub_box, mu, rc, p = args
    return _calc_pbc_hic_cpu_serial(sub_xyz, sub_box, mu, rc, p)


def _calc_HiC_from_traj_array(traj, mu, rc, p):
    # print('Computing probability of contact versus contour distance')
    pol_size = traj.shape[1]
    Prob = np.zeros((pol_size, pol_size))
    for ii, snapshot in enumerate(traj):
        Prob += _calc_prob(snapshot, mu, rc, p)
    Prob = Prob / (ii + 1)
    return Prob


# =============================================================================
# Helper Functions (Fixed Precision and Logic)
# =============================================================================


def _mic_distance(pos, box):
    """
    Calculates Euclidean distance matrix using Minimum Image Convention.
    Uses float64 for intermediate calculation to avoid precision artifacts.
    """
    # Ensure float64
    pos = pos.astype(np.float64)
    box = box.astype(np.float64)
    L = np.diag(box)

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]

    # MIC correction
    # diff / L can be sensitive if L is small and diff is large
    diff -= L * np.round(diff / L)

    return np.linalg.norm(diff, axis=-1)


def _calc_pbc_hic_cpu_serial(traj, boxes, mu, rc, p):
    n_beads = traj.shape[1]
    total_prob = np.zeros((n_beads, n_beads), dtype=np.float64)  # Accumulate in double

    for i in range(traj.shape[0]):
        # 1. Calculate Distance with PBC
        r = _mic_distance(traj[i], boxes[i])

        if p is not None:
            # Safe division for power law
            with np.errstate(divide="ignore", invalid="ignore"):
                term_power = 0.5 * (rc / r) ** p
            term_power[r == 0] = 0.0  # Fix diagonal

            prob = np.where(r <= rc, 0.5 * (1 + np.tanh(mu * (rc - r))), term_power)
        else:
            prob = 0.5 * (1 + np.tanh(mu * (rc - r)))

        np.fill_diagonal(prob, 0.0)
        total_prob += prob

    return (total_prob / traj.shape[0]).astype(np.float32)


# def _calc_prob(data, mu, rc, p):
#     r = distance.cdist(data, data, 'euclidean')
#     f = np.where(
#         r <= rc,
#         0.5 * (1 + np.tanh(mu * (rc - r))),
#         0.5 * (rc / r)**p
#     )
#     return f
