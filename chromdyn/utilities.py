#  * --------------------------------------------------------------------------- *
#  *                                  chromdyn                                   *
#  * --------------------------------------------------------------------------- *
#  * This is part of the chromdyn simulation toolkit released under MIT License. *
#  *                                                                             *
#  * Author: Sumitabha Brahmachari                                               *
#  * --------------------------------------------------------------------------- *

import numpy as np
import random
import os
import logging


class LogManager:
    def __init__(
        self,
        log_level=logging.INFO,
        log_format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        log_file=None,
    ):
        """
        Initialize LoggerManager with log level, format, and optional file logging.

        Args:
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
            log_format (str): Logging format.
            log_file (str, optional): Path to log file. If None, logs only to console.
        """
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self._owned_handlers = []

    def get_logger(self, name, console=True, *, isolated=False):
        """Get a module logger, or an independently owned simulation logger.

        Module configuration is idempotent and preserves application handlers.
        File-backed/isolated loggers are not placed in logging's global registry,
        so simulations with the same module name cannot redirect each other.
        """
        isolated = isolated or self.log_file is not None
        if isolated:
            logger = logging.Logger(name, self.log_level)
            logger.propagate = False
        else:
            logger = logging.getLogger(name)
            if logger.handlers:
                return logger
            if logger.level == logging.NOTSET:
                logger.setLevel(self.log_level)
        handlers = []
        if console and (isolated or not logger.hasHandlers()):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handlers.append(handler)
        if self.log_file is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            handler = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handlers.append(handler)
        if isolated and not handlers:
            handlers.append(logging.NullHandler())
        for handler in handlers:
            handler.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(handler)
            self._owned_handlers.append((logger, handler))
        return logger

    def close(self):
        """Close only handlers created by this manager; repeated calls are safe."""
        for logger, handler in self._owned_handlers:
            logger.removeHandler(handler)
            handler.close()
        self._owned_handlers.clear()

    @staticmethod
    def log_header(logger, title, width=100, char="="):
        """
        Prints a formatted header in logs for better readability.

        Args:
            logger (logging.Logger): Logger instance to use.
            title (str): Header title to display.
            width (int): Total width of the header (default 100).
            char (str): Character to use for border lines (default '=').
        """
        logger.info(char * width)
        logger.info(f"{title:^{width}}")  # Center the title
        logger.info(char * width)


class config_generator:
    def __init__(self, max_restarts: int = 5000):
        self.max_restarts = max_restarts
        self.valid_modes = ["randomwalk", "saw3d", "random"]

    def get_config(self, mode: str, num_steps: int, **kwargs):
        step_size = kwargs.get("step_size", 1)
        Rc = kwargs.get("Rc", np.sqrt(num_steps))

        if mode.lower() == "saw3d":
            try:
                return self.get_pos_3Dsaw(num_steps, step_size)
            except RuntimeError as e:
                path, rw_msg = self.get_pos_3Drandom_walk(num_steps, step_size)
                msg = f"{e} {rw_msg}"
                return path, msg

        elif mode.lower() == "randomwalk":
            # Generate a random walk
            return self.get_pos_3Drandom_walk(num_steps, step_size)

        elif mode.lower() == "random":
            pos = np.random.random(size=(num_steps, 3)) * Rc
            msg = f"Random configuration created with max excursion {Rc:2f}. Position shape: {pos.shape}."
            return pos, msg

        else:
            path, _ = self.get_pos_3Drandom_walk(num_steps, step_size)
            msg = f"Invalid mode '{mode}'. Defaulting to random walk. Position shape: {path.shape}."
            return path, msg

    def get_pos_3Dsaw(self, num_steps, step_size):
        """
        Generate a 3D self-avoiding walk on a cubic lattice with restarts.

        Args:
            num_steps (int): Desired length of the walk.
            max_restarts (int): Maximum number of full retries if stuck.
            verbose (bool): Print progress info.

        Returns:
            path (np.ndarray): Array of shape (num_steps, 3) with 3D walk positions.
        """
        directions = [
            np.array([step_size, 0, 0]),
            np.array([-step_size, 0, 0]),
            np.array([0, step_size, 0]),
            np.array([0, -step_size, 0]),
            np.array([0, 0, step_size]),
            np.array([0, 0, -step_size]),
        ]

        for attempt in range(1, self.max_restarts + 1):
            position = np.array([0, 0, 0])
            visited = set()
            visited.add(tuple(position))
            path = [position.copy()]
            success = True

            for step in range(1, num_steps):
                # List all valid next positions
                valid_moves = []
                for d in directions:
                    candidate = tuple(position + d)
                    if candidate not in visited:
                        valid_moves.append(d)

                if not valid_moves:
                    success = False
                    # if verbose:
                    #     self.logger.info(f"[Attempt {attempt}] Stuck at step {step}, restarting...")
                    break  # restart the whole walk

                move = random.choice(valid_moves)
                position += move
                visited.add(tuple(position))
                path.append(position.copy())

            if success:
                path = np.array(path)
                msg = f"3D SAW created after {attempt} attempt(s). Position shape: {path.shape}"
                return path, msg
        raise RuntimeError(
            f"Failed to generate a self-avoiding walk after {self.max_restarts} attempts."
        )

    def get_pos_3Drandom_walk(self, num_steps, step_size):
        # Generate random directions on the unit sphere
        directions = np.random.normal(size=(num_steps, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
        steps = directions * step_size  # Scale to step size
        positions = np.cumsum(steps, axis=0)  # Cumulative sum for path
        msg = f"Random walk created. Position shape: {positions.shape}"
        return positions, msg  # Shape: (num_steps, 3)
