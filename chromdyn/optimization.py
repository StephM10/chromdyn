#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

import numpy as np
import h5py
from pathlib import Path
from .hic_utils import HiCManager

hicman = HiCManager()


class EnergyLandscapeOptimizer:

    def __init__(
        self,
        method: str = "adam",
        eta: float = 0.01,
        beta1: float = 0.7,
        beta2: float = 0.9999,
        epsilon: float = 1e-8,
        it: int = 1,
        scheduler: str = "none",
        scheduler_decay: float = 0.1,
        scheduler_step: int = 100,
        scheduler_eta_min: float = 0.001,
        scheduler_T_max: int = 120,
        sgd_batch_fraction: float = 0.5,
        save_dir: str = "opt_params",
    ):
        """
        Initializes the Energy Landscape Optimizer with given hyperparameters and learning rate scheduler.

        Parameters:
            mu, rc: Model hyperparameters.
            method: Optimizer method, e.g., "adam", "nadam", "rmsprop", "adagrad", "sgd", "gd".
                - "gd": Vanilla (batch) gradient descent using the full gradient.
                - "sgd": Stochastic gradient descent with mini-batch sampling of matrix entries.
            eta: Initial learning rate.
            beta1, beta2, epsilon: Adam-related hyperparameters.
            it: Initial iteration counter.
            error_pca_weight: Weight for regularizing error using PCA.
            scheduler: Type of learning rate scheduler ("none", "step", "cosine", or "exponential").
            scheduler_decay: Decay factor (used as lambda for exponential decay, or decay factor for step decay).
            scheduler_step: Number of iterations per step decay.
            scheduler_eta_min: Minimum learning rate for the cosine scheduler.
            scheduler_T_max: Total iterations for cosine annealing.
            sgd_batch_fraction: Fraction of matrix entries to sample per SGD step (0, 1]. Default 0.5.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta0 = eta  # Store the initial learning rate
        self.eta = eta  # Current learning rate
        self.t = int(it)
        self.method = method.lower()
        self.opt_params = None
        self.phi_exp = None  # Experimental Hi-C data
        self.force_field = None
        self.updated_force_field = None
        self.error = None
        self.sgd_batch_fraction = sgd_batch_fraction
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # Learning rate scheduler parameters
        self.scheduler = scheduler.lower()  # "none", "step", "cosine", or "exponential"
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.scheduler_eta_min = scheduler_eta_min
        self.scheduler_T_max = scheduler_T_max

    def update_learning_rate(self) -> None:
        """Updates the learning rate based on the selected scheduler."""
        if self.scheduler == "step":
            # Step decay: Every scheduler_step iterations, multiply by scheduler_decay.
            factor = self.scheduler_decay ** (self.t // self.scheduler_step)
            self.eta = self.eta0 * factor
        elif self.scheduler == "cosine":
            # Cosine annealing: Gradually decay the learning rate with a cosine schedule.
            self.eta = self.scheduler_eta_min + 0.5 * (
                self.eta0 - self.scheduler_eta_min
            ) * (1 + np.cos(np.pi * self.t / self.scheduler_T_max))
        elif self.scheduler == "exponential":
            # Exponential decay: Learning rate decays as: eta_t = eta0 * exp(-lambda * t)
            self.eta = self.eta0 * np.exp(-self.scheduler_decay * self.t)
        elif self.scheduler == "none":
            # If scheduler is "none", self.eta remains unchanged.
            self.eta = self.eta0

    def load_HiC(
        self,
        hic_file: str,
        cutoff_low: float = 0.0,
        cutoff_high: float = 1.0,
        neighbors: int = 0,
        filter: str = "None",
    ) -> None:
        """
        Loads the Hi-C matrix from a text file, applies cutoffs, and initializes optimization parameters.
        """
        if not hic_file.endswith(".txt"):
            raise ValueError(
                "Input Hi-C file should be a TXT file that can be handled by np.loadtxt."
            )

        hic_mat = np.loadtxt(hic_file)

        if not self.is_symmetric(hic_mat):
            raise ValueError("Experimental HiC input is NOT symmetric.")

        if filter != "None":
            hic_mat = hicman.filter_matrix(hic_mat, method=filter)
        # Apply cutoffs to remove noise
        hic_mat = np.clip(hic_mat, a_min=cutoff_low, a_max=cutoff_high)

        # Remove neighbor interactions within the given range
        neighbor_mask = (
            np.abs(np.subtract.outer(np.arange(len(hic_mat)), np.arange(len(hic_mat))))
            <= neighbors
        )
        hic_mat[neighbor_mask] = 0.0
        # hic_mat[hic_mat>cutoff_high]=0.0
        # hic_mat[hic_mat<cutoff_low]=0.0
        self.phi_exp = hic_mat
        self.mask = hic_mat != 0.0
        if self.opt_params is None:
            self.init_optimization_params()

    @staticmethod
    def is_symmetric(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if a matrix is symmetric."""
        return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

    def set_optimization_params(self, params_h5file: str) -> None:
        self.opt_params = {}

        with h5py.File(params_h5file, "r") as h5file:
            for key in h5file.keys():
                data = h5file[key][()]
                # Decode bytes if necessary
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                self.opt_params[key] = data

    def save_optimization_params(self) -> None:
        with h5py.File(
            str(Path(self.save_dir) / f"{self.method}_params_{self.t}.h5"), "w"
        ) as h5file:
            for key, value in self.opt_params.items():
                h5file.create_dataset(key, data=value)

    def init_optimization_params(self) -> None:
        """Initializes optimization parameters for different optimizers."""
        self.opt_params = {}
        shape = self.phi_exp.shape

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] = np.zeros(shape)
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "rmsprop":
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "adagrad":
            self.opt_params["G_dw"] = np.zeros(shape)

    def update_step(self, grad: np.ndarray, lambda_t: np.ndarray) -> np.ndarray:
        """Performs an optimization step based on the selected method, updating the learning rate if a scheduler is used."""
        # Update the learning rate based on the scheduler
        self.update_learning_rate()

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] *= self.beta1
            self.opt_params["m_dw"] += (1 - self.beta1) * grad
            self.opt_params["v_dw"] *= self.beta2
            self.opt_params["v_dw"] += (1 - self.beta2) * (grad**2)

            m_dw_corr = self.opt_params["m_dw"] / (1 - self.beta1**self.t)
            v_dw_corr = self.opt_params["v_dw"] / (1 - self.beta2**self.t)

            if self.method == "nadam":
                lookahead_gradient = (1 - self.beta1) * grad / (1 - self.beta1**self.t)
                m_dw_corr += lookahead_gradient

            w = lambda_t - self.eta * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)

        elif self.method == "rmsprop":
            self.opt_params["v_dw"] *= self.beta1
            self.opt_params["v_dw"] += (1 - self.beta1) * (grad**2)
            w = lambda_t - self.eta * grad / (
                np.sqrt(self.opt_params["v_dw"]) + self.epsilon
            )

        elif self.method == "adagrad":
            self.opt_params["G_dw"] += grad**2
            w = lambda_t - self.eta * grad / (
                np.sqrt(self.opt_params["G_dw"]) + self.epsilon
            )

        elif self.method == "gd":
            w = lambda_t - self.eta * grad

        elif self.method == "sgd":
            # True stochastic gradient descent: randomly sample a fraction of
            # upper-triangular gradient entries, then symmetrize.
            rand_mask = np.random.rand(*grad.shape) < self.sgd_batch_fraction
            rand_mask = np.triu(rand_mask, k=1)
            rand_mask = rand_mask | rand_mask.T
            np.fill_diagonal(rand_mask, False)
            # Scale by 1/fraction so the expected value equals the full gradient
            stochastic_grad = grad * rand_mask / self.sgd_batch_fraction
            w = lambda_t - self.eta * stochastic_grad

        self.save_optimization_params()
        self.t += 1
        return w

    def get_error_gradient(self, phi_sim: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the optimization objective."""
        gt = self.phi_exp - phi_sim
        np.fill_diagonal(gt, 0.0)
        # gt -= np.diagflat(np.diag(gt, k=1), k=1)
        gt = np.triu(gt) + np.triu(gt).T  # Ensure symmetry
        return gt

    def set_opt_method(self, method: str) -> None:
        self.method = method.lower()

    def get_updated_params(
        self, lambda_t: np.ndarray, phi_sim: np.ndarray
    ) -> np.ndarray:
        """Computes and updates the force field from the given file."""
        phi_sim *= self.mask
        grad = self.get_error_gradient(phi_sim)
        updated_lambda = self.update_step(grad, lambda_t)
        return updated_lambda

    def get_error(self, phi_sim: np.ndarray) -> float:
        self.error = np.sum(
            np.abs(np.triu(phi_sim, k=2) - np.triu(self.phi_exp, k=2))
        ) / np.sum(np.triu(self.phi_exp, k=2))
        return self.error
