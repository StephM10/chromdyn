"""
Visualization module for ChromDyn.
This module is an optional dependency. It requires 'matplotlib' to be installed.
"""

import warnings
import numpy as np
import itertools

try:
    import matplotlib.pyplot as plt

    # import matplotlib.colors as mcolors
    # from matplotlib import cm
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    MATPLOTLIB_AVAILABLE = True

except ImportError as e:
    # catch specific ImportError to know if matplotlib or mpl_toolkits is missing
    warnings.warn(f"Visualization dependencies not available: {e}")
    MATPLOTLIB_AVAILABLE = False

    # define empty variables to prevent IDE static checks from Error
    plt = None
    Axes3D = None
    FuncAnimation = None
    PillowWriter = None
    FFMpegWriter = None
    Line3DCollection = None


# Helper function to raise error if matplotlib is missing
def _check_matplotlib():
    """Helper function to raise error if matplotlib is missing."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Visualization features require 'matplotlib'. "
            "Please install it via 'pip install matplotlib'."
        )


# Plotting


def _draw_pbc_box(
    ax,
    box_a,
    center=np.array([0, 0, 0]),
    color="gray",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
):
    """Helper function to draw a cubic PBC box centered at `center`."""
    half_a = box_a / 2.0
    min_coords = center - half_a
    max_coords = center + half_a

    corners = np.array(
        [
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )


def recenter_coordinates_v3(polymer_coords_list, box_vectors):
    """
    Recenter a list of polymer chain coordinates based on generic box vectors.
    The system's COM is moved to the geometric center of the box defined by box_vectors.

    Args:
        polymer_coords_list (list of np.ndarray): Coordinates.
        box_vectors (np.ndarray): 3x3 array [[ax, ay, az], [bx, by, bz], [cx, cy, cz]].

    Returns:
        list of np.ndarray: Recentered coordinates.
    """

    _check_matplotlib()

    if not polymer_coords_list or not any(
        chain.size > 0 for chain in polymer_coords_list
    ):
        return polymer_coords_list

    # 1. Calculate System COM
    all_coords_flat = np.vstack(
        [
            chain
            for chain in polymer_coords_list
            if chain.ndim == 2 and chain.shape[1] == 3 and chain.size > 0
        ]
    )
    if all_coords_flat.size == 0:
        return polymer_coords_list

    current_com = np.mean(all_coords_flat, axis=0)

    # 2. Calculate Box Geometric Center
    # Center = Origin + 0.5 * (vec_a + vec_b + vec_c)
    # Assuming origin is (0,0,0)
    box_center = 0.5 * np.sum(box_vectors, axis=0)

    # 3. Shift
    shift_vector = box_center - current_com

    recentered_coords_list = []
    for chain_coords in polymer_coords_list:
        if (
            chain_coords.ndim == 2
            and chain_coords.shape[1] == 3
            and chain_coords.size > 0
        ):
            recentered_coords_list.append(chain_coords + shift_vector)
        else:
            recentered_coords_list.append(chain_coords)

    return recentered_coords_list


def _draw_generic_box(ax, box_vectors, color="k", alpha=0.5, linewidth=1.0):
    """
    Draws a wireframe parallelepiped defined by box_vectors starting at (0,0,0).
    """
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = box_vectors[0]  # a
    v2 = box_vectors[1]  # b
    v3 = box_vectors[2]  # c

    # The 8 corners of the box
    # 0: origin
    # 1: a
    # 2: b
    # 3: c
    # 4: a+b
    # 5: a+c
    # 6: b+c
    # 7: a+b+c
    corners = np.array([v0, v1, v2, v3, v1 + v2, v1 + v3, v2 + v3, v1 + v2 + v3])

    # Define the 12 edges by connecting corner indices
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],  # Origin to axes
        [1, 4],
        [1, 5],  # From a
        [2, 4],
        [2, 6],  # From b
        [3, 5],
        [3, 6],  # From c
        [4, 7],
        [5, 7],
        [6, 7],  # To far corner
    ]

    lines = []
    for start_idx, end_idx in edges:
        lines.append([corners[start_idx], corners[end_idx]])

    # Plot using Line3DCollection for efficiency
    lc = Line3DCollection(lines, colors=color, alpha=alpha, linewidths=linewidth)
    ax.add_collection3d(lc)

    return corners  # Return corners to help set axis limits


# Universal visualization function with optional PBC(draw box)
def visualize(
    traj,
    select_frame=0,
    axis_limits=None,
    colors=None,
    output_name=None,
    isring=False,
    r=None,
    recenter=False,
    color_mode="chain",
    types=None,
    PBC=False,
):
    """
    Universal visualization function for polymer chains with optional PBC box.

    Args:
        traj: Trajectory object containing coordinate and topology data.
        select_frame (int): Frame index to visualize (default: 0).
        axis_limits (tuple): Manual axis limits as (x_min, x_max, y_min, y_max, z_min, z_max).
        colors (list): Custom colors for chains.
        output_name (str): If provided, save the plot to this filename instead of displaying.
        isring (bool): If True, connect the last bead to the first bead (default: False).
        r (float): Physical radius of beads in nm for size calculation.
        recenter (bool): If True, recenter coordinates (default: True).
        color_mode (str): 'chain' (default) or 'type' for coloring scheme.
        types (list): Custom type sequence to override trajectory types.
        PBC (bool): If True, draw periodic boundary box and use box-based recentering (default: False).
    """

    _check_matplotlib()

    if PBC:
        recenter = True
        print("PBC is enabled, automatically recentering.")

    # --- 1. Data & Topology Check ---
    if not hasattr(traj, "topology") or traj.topology is None:
        print("Error: Topology not found in trajectory.")
        return

    chain_info_list = traj.chain_info
    n_chains = len(chain_info_list)
    bead_counts = [count for _, count in chain_info_list]

    # Calculate indices
    cumulative_indices = np.cumsum([0] + bead_counts)
    chain_selections = [
        np.arange(start, end)
        for start, end in zip(cumulative_indices[:-1], cumulative_indices[1:])
    ]

    # --- 2. Load Coordinates (With Fix) ---
    polymer_coords_orig = []
    try:
        for sel in chain_selections:
            # [CRITICAL FIX]: Convert numpy array 'sel' to list using .tolist()
            # This prevents "truth value of an array" errors inside traj.xyz
            sel_list = sel.tolist()

            # xyz returns (n_frames, n_beads, 3), take [0] for single frame
            data = traj.xyz(
                frames=[select_frame, select_frame + 1, 1], bead_selection=sel_list
            )

            if data.shape[0] > 0:
                polymer_coords_orig.append(np.nan_to_num(data[0]))
            else:
                polymer_coords_orig.append(np.array([]))

    except Exception as e:
        print(f"Error loading coordinates at frame {select_frame}: {e}")
        return

    # Check if we got data
    if not polymer_coords_orig or all(c.size == 0 for c in polymer_coords_orig):
        print(f"Error: No valid coordinate data found for frame {select_frame}.")
        return

    if PBC:
        # --- 3. Load Box Vectors ---
        box_vectors = None
        if hasattr(traj, "box_vectors") and (traj.box_vectors is not None):
            try:
                box_vectors = traj.box_vectors[select_frame]
            except Exception:
                pass  # Fallback below

        if box_vectors is None:
            print("Warning: No box vectors found. Assuming 100nm cube.")
            box_vectors = np.eye(3) * 100.0

    # --- 4. Recenter ---
    if recenter:
        print("Recentering coordinates...")
        polymer_coords = recenter_coordinates_v3(polymer_coords_orig, box_vectors)
    else:
        polymer_coords = polymer_coords_orig

    # --- 5. Color Setup ---
    chain_colors_list = []  # For 'chain' mode
    bead_colors_list = []  # For 'type' mode (list of color arrays)
    type_legend_handles = {}

    # Mode: Type
    if color_mode == "type":
        # Get all types
        types_seq = getattr(traj, "chrom_seq", getattr(traj, "types", None))
        # if types are provided, use them
        if types is not None:
            types_seq = types
            print(f"Using provided types: {types_seq}")
        if types_seq is None:
            print(
                "Warning: 'type' mode requested but no types found. Switching to 'chain'."
            )
            color_mode = "chain"
        else:
            print(f"Using types from trajectory: {types_seq}")
            unique_types = sorted(list(set(types_seq)))
            cmap = plt.get_cmap("tab10")
            type_map = {t: cmap(i % 10) for i, t in enumerate(unique_types)}

            # Convert all types to colors once
            all_bead_colors = np.array([type_map[t] for t in types_seq])

            # Slice colors for each chain
            for sel in chain_selections:
                bead_colors_list.append(all_bead_colors[sel])

            # Create Legend Handles
            for t, c in type_map.items():
                type_legend_handles[t] = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    markersize=10,
                    label=f"Type {t}",
                )

    # Mode: Chain
    if color_mode == "chain":
        if colors is None:
            cmap = plt.get_cmap("tab10")
            chain_colors_list = [cmap(i % 10) for i in range(n_chains)]
        else:
            chain_colors_list = colors

    # --- 6. Plotting ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter_collections = []

    for i, chain in enumerate(polymer_coords):
        if chain.size == 0:
            continue

        xs, ys, zs = chain[:, 0], chain[:, 1], chain[:, 2]

        # Determine Colors and Plot arguments
        if color_mode == "chain":
            c_val = chain_colors_list[i]  # Single RGBA tuple
            line_c = c_val
            line_a = 0.7
            label_val = f"Chain {i+1}" if n_chains <= 10 else ""

            # [FIX]: Use 'color' for single color to avoid warning
            kwargs_scatter = {"color": c_val}
        else:
            c_val = bead_colors_list[i]  # Array of RGBA tuples
            line_c = "gray"
            line_a = 0.3
            label_val = ""

            # [FIX]: Use 'c' for color array
            kwargs_scatter = {"c": c_val}

        # Draw Lines
        if isring:
            ax.plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                np.append(zs, zs[0]),
                color=line_c,
                alpha=line_a,
                linewidth=2.0,
            )
        else:
            ax.plot(xs, ys, zs, color=line_c, alpha=line_a, linewidth=2.0)

        # Draw Beads
        initial_s = 1 if r is not None else 20
        sc = ax.scatter(
            xs, ys, zs, alpha=0.8, s=initial_s, label=label_val, **kwargs_scatter
        )
        scatter_collections.append(sc)

    if PBC:
        # --- 7. Draw PBC Box ---
        box_corners = _draw_generic_box(ax, box_vectors)

        # Labels & Legend
        ax.set_xlabel(r"X ($\sigma$)")
        ax.set_ylabel(r"Y ($\sigma$)")
        ax.set_zlabel(r"Z ($\sigma$)")

        if color_mode == "type":
            ax.legend(handles=type_legend_handles.values(), loc="best")
        elif color_mode == "chain" and n_chains <= 10:
            ax.legend(loc="best")

        title_str = f"Frame {select_frame} (c:{color_mode})"
        if recenter:
            title_str += " (Recentered)"
        ax.set_title(title_str)

    # --- 8. View & Limits ---
    if r is not None:
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=-45)
    else:
        ax.set_proj_type("persp")

    if axis_limits:
        x_min, x_max, y_min, y_max, z_min, z_max = axis_limits
    elif PBC:
        all_poly = np.vstack([c for c in polymer_coords if c.size > 0])
        all_points = np.vstack([all_poly, box_corners])
        min_ext = np.min(all_points, axis=0)
        max_ext = np.max(all_points, axis=0)
        center = (min_ext + max_ext) / 2.0
        span = max_ext - min_ext
        max_span = np.max(span) if np.max(span) > 0 else 10.0
        buffer = max_span * 0.55
        x_min, x_max = center[0] - buffer, center[0] + buffer
        y_min, y_max = center[1] - buffer, center[1] + buffer
        z_min, z_max = center[2] - buffer, center[2] + buffer
    else:
        all_poly = np.vstack([c for c in polymer_coords if c.size > 0])
        min_ext = np.min(all_poly, axis=0)
        max_ext = np.max(all_poly, axis=0)
        center = (min_ext + max_ext) / 2.0
        span = max_ext - min_ext
        max_span = np.max(span) if np.max(span) > 0 else 10.0
        buffer = max_span * 0.55
        x_min, x_max = center[0] - buffer, center[0] + buffer
        y_min, y_max = center[1] - buffer, center[1] + buffer
        z_min, z_max = center[2] - buffer, center[2] + buffer

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])

    # --- 9. Update Scatter Size ---
    if r is not None:
        fig.canvas.draw()
        data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        bbox = ax.get_window_extent()
        if bbox and data_range > 0:
            points_per_unit = (bbox.width * 72 / fig.get_dpi()) / data_range
            s_phys = np.clip(np.pi * ((r * points_per_unit) ** 2), 0.01, 50000)
            for sc in scatter_collections:
                sc.set_sizes(np.full(len(sc.get_offsets()), s_phys))

    # --- 10. Output & Display ---
    if output_name:
        plt.savefig(output_name, dpi=300)
        print(f"Plot saved to {output_name}")
        plt.close(fig)
    else:
        plt.show()


def visualize_animation(
    traj,
    start_frame=0,
    end_frame=None,
    fps=20,
    axis_limits=None,
    colors=None,
    output_name=None,
    isring=False,
    r=None,
    recenter=True,
    color_mode="chain",
    types=None,
    PBC=False,
):
    """
    Universal animation visualization function for polymer chains with optional PBC box.

    Args:
        traj: Trajectory object containing coordinate and topology data.
        start_frame (int): Starting frame index (default: 0).
        end_frame (int): Ending frame index (default: None, uses all frames).
        fps (int): Frames per second for animation (default: 20).
        axis_limits (tuple): Manual axis limits as (x_min, x_max, y_min, y_max, z_min, z_max).
        colors (list): Custom colors for chains.
        output_name (str): If provided, save animation to this filename.
        isring (bool): If True, connect the last bead to the first bead (default: False).
        r (float): Physical radius of beads in nm for size calculation.
        recenter (bool): If True, recenter coordinates (default: True).
        color_mode (str): 'chain' (default) or 'type' for coloring scheme.
        types (list): Custom type sequence to override trajectory types.
        PBC (bool): If True, draw periodic boundary box and use box-based recentering (default: False).

    Returns:
        Animation object if no output_name is provided.
    """
    _check_matplotlib()

    if PBC:
        recenter = True
        print("PBC is True, recentering coordinates automatically.")

    # --- 1. Data Loading ---
    if not hasattr(traj, "topology") or traj.topology is None:
        print("Error: Topology not found.")
        return

    chain_info = traj.chain_info
    n_chains = len(chain_info)
    bead_counts = [c[1] for c in chain_info]
    cumulative_indices = np.cumsum([0] + bead_counts)
    chain_selections = [
        np.arange(s, e) for s, e in zip(cumulative_indices[:-1], cumulative_indices[1:])
    ]

    total_frames = traj.n_frames
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame < 0:
        start_frame = 0

    print(f"Loading frames {start_frame} to {end_frame}...")

    # --- 2. Load Coordinates (Defensive Fix) ---
    try:
        polymer_coords_all_chains_orig = []
        for sel in chain_selections:
            # [FIX]: Convert to list to prevent 'truth value of array' error
            sel_list = sel.tolist()
            data = traj.xyz(frames=[start_frame, end_frame, 1], bead_selection=sel_list)
            polymer_coords_all_chains_orig.append(np.nan_to_num(data))
    except Exception as e:
        print(f"Error calling traj.xyz: {e}")
        return

    if not polymer_coords_all_chains_orig:
        return
    num_anim_frames = end_frame - start_frame

    # --- 3. Load Box Vectors (if PBC)---
    if PBC:
        if hasattr(traj, "box_vectors") and (traj.box_vectors is not None):
            box_vectors_range = traj.box_vectors[start_frame:end_frame]
        else:
            box_vectors_range = np.tile(np.eye(3) * 100.0, (num_anim_frames, 1, 1))

    # --- 4. Color Setup ---
    chain_colors_list = []
    bead_colors_list = []
    type_legend_handles = {}

    if color_mode == "type":
        types_seq = getattr(traj, "chrom_seq", getattr(traj, "types", None))
        # if types are provided, use them
        if types is not None:
            types_seq = types
        if types_seq is None:
            color_mode = "chain"
        else:
            unique_types = sorted(list(set(types_seq)))
            cmap = plt.get_cmap("tab10")
            type_map = {t: cmap(i % 10) for i, t in enumerate(unique_types)}
            all_bead_colors = np.array([type_map[t] for t in types_seq])
            for sel in chain_selections:
                bead_colors_list.append(all_bead_colors[sel])
            for t, c in type_map.items():
                type_legend_handles[t] = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    markersize=10,
                    label=f"Type {t}",
                )

    if color_mode == "chain":
        if colors is None:
            cmap = plt.get_cmap("tab10")
            chain_colors_list = [cmap(i % 10) for i in range(n_chains)]
        else:
            chain_colors_list = colors

    # --- 5. Process Frames ---
    all_frames_processed = []
    for i in range(num_anim_frames):
        frame_coords = [chain_data[i] for chain_data in polymer_coords_all_chains_orig]

        if PBC and recenter:
            box = box_vectors_range[i]
            processed = recenter_coordinates_v3(frame_coords, box)
            all_frames_processed.append(processed)
        elif recenter:
            # Simple recentering - move COM to origin
            all_coords_flat = np.vstack(
                [
                    chain
                    for chain in frame_coords
                    if chain.ndim == 2 and chain.shape[1] == 3 and chain.size > 0
                ]
            )
            if all_coords_flat.size > 0:
                current_com = np.mean(all_coords_flat, axis=0)
                processed = [
                    chain - current_com if chain.size > 0 else chain
                    for chain in frame_coords
                ]
                all_frames_processed.append(processed)
            else:
                all_frames_processed.append(frame_coords)
        else:
            all_frames_processed.append(frame_coords)

    # --- 6. Limits ---
    if axis_limits:
        x_min, x_max, y_min, y_max, z_min, z_max = axis_limits
    elif PBC:
        # Fast auto-limit using Frame 0 Box
        v = box_vectors_range[0]
        center = 0.5 * np.sum(v, axis=0)
        max_span = max(np.linalg.norm(v[0]), np.linalg.norm(v[1]), np.linalg.norm(v[2]))
        buffer = max_span * 0.6
        x_min, x_max = center[0] - buffer, center[0] + buffer
        y_min, y_max = center[1] - buffer, center[1] + buffer
        z_min, z_max = center[2] - buffer, center[2] + buffer
    else:
        # Auto-limit using all frames
        all_points = []
        for frame_coords in all_frames_processed:
            for chain in frame_coords:
                if chain.size > 0:
                    all_points.append(chain)

        if all_points:
            all_points = np.vstack(all_points)
            min_ext = np.min(all_points, axis=0)
            max_ext = np.max(all_points, axis=0)
            center = (min_ext + max_ext) / 2.0
            span = max_ext - min_ext
            max_span = np.max(span) if np.max(span) > 0 else 10.0
            buffer = max_span * 0.6
            x_min, x_max = center[0] - buffer, center[0] + buffer
            y_min, y_max = center[1] - buffer, center[1] + buffer
            z_min, z_max = center[2] - buffer, center[2] + buffer
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -50, 50

    # --- 7. Setup Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if r is not None:
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=-45)
    else:
        ax.set_proj_type("persp")
    ax.set_box_aspect([1, 1, 1])

    scatters = []
    lines = []
    initial_s = 1 if r is not None else 20

    for i in range(n_chains):
        if color_mode == "chain":
            c_val = chain_colors_list[i]
            line_c = c_val
            line_a = 0.5
            # [FIX] Use 'color' for single RGBA
            sc = ax.scatter([], [], [], color=c_val, alpha=0.7, s=initial_s)
        else:
            # Type mode: colors set in update
            line_c = "gray"
            line_a = 0.3
            # [FIX] Initialize with empty, color set later
            sc = ax.scatter([], [], [], alpha=0.7, s=initial_s)

        (ln,) = ax.plot([], [], [], color=line_c, alpha=line_a, linewidth=2.0)
        scatters.append(sc)
        lines.append(ln)

    if PBC:
        _draw_generic_box(ax, box_vectors_range[0])

    if color_mode == "type":
        ax.legend(handles=type_legend_handles.values(), loc="best")

    # Size calculation
    if r is not None:
        fig.canvas.draw()
        data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        bbox = ax.get_window_extent()
        if bbox and data_range > 0:
            ppt = (bbox.width * 72 / fig.get_dpi()) / data_range
            s_phys = np.clip(np.pi * ((r * ppt) ** 2), 0.01, 50000)
            for sc in scatters:
                sc._sizes = np.array([s_phys])

    # --- 8. Update ---
    def update(frame_idx):
        coords = all_frames_processed[frame_idx]
        artists = []
        for i, (sc, ln) in enumerate(zip(scatters, lines)):
            chain = coords[i]
            if chain.size > 0:
                sc._offsets3d = (chain[:, 0], chain[:, 1], chain[:, 2])

                # [FIX] Update colors for Type mode
                if color_mode == "type":
                    sc.set_facecolors(bead_colors_list[i])
                    sc.set_edgecolors(bead_colors_list[i])

                xs, ys, zs = chain[:, 0], chain[:, 1], chain[:, 2]
                if isring:
                    xs = np.append(xs, xs[0])
                    ys = np.append(ys, ys[0])
                    zs = np.append(zs, zs[0])
                ln.set_data(xs, ys)
                ln.set_3d_properties(zs)
            else:
                sc._offsets3d = ([], [], [])
                ln.set_data([], [])
                ln.set_3d_properties([])
            artists.extend([sc, ln])
        ax.set_title(f"Frame {start_frame + frame_idx} ({color_mode})")
        return artists

    print("Generating animation...")
    anim = FuncAnimation(
        fig, update, frames=num_anim_frames, interval=1000 / fps, blit=False
    )

    if output_name:
        writer = (
            FFMpegWriter(fps=fps)
            if output_name.endswith(".mp4")
            else PillowWriter(fps=fps)
        )
        anim.save(output_name, writer=writer, dpi=300)
        print(f"Saved to {output_name}")
        plt.close(fig)
    else:
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            plt.show()
        return anim


# This function actually relies on the topology format inside cndb files
# if we use a different format, we need to modify this function
def visualize_pbc_images(
    traj,
    select_frame=0,
    n_layers=1,
    image_alpha=0.15,
    image_style="scatter",
    axis_limits=None,
    colors=None,
    output_name=None,
    isring=False,
    r=None,
    recenter=True,
    color_mode="chain",
    types=None,
):
    """
    Visualize central polymer AND periodic images with correct physical sizing.
    """
    _check_matplotlib()

    # --- 1. Data Loading ---
    if not hasattr(traj, "topology") or traj.topology is None:
        print("Error: Topology not found.")
        return

    # Check Box
    box_vectors = None
    if hasattr(traj, "box_vectors") and (traj.box_vectors is not None):
        try:
            box_vectors = traj.box_vectors[select_frame]
        except Exception:
            pass
    if box_vectors is None:
        print("Error: No box vectors.")
        return

    # Load Coords
    chain_info = traj.chain_info
    bead_counts = [c[1] for c in chain_info]
    cumulative_indices = np.cumsum([0] + bead_counts)
    chain_selections = [
        np.arange(s, e).tolist()
        for s, e in zip(cumulative_indices[:-1], cumulative_indices[1:])
    ]

    polymer_coords_orig = []
    try:
        for sel in chain_selections:
            data = traj.xyz(
                frames=[select_frame, select_frame + 1, 1], bead_selection=sel
            )
            if data.shape[0] > 0:
                polymer_coords_orig.append(np.nan_to_num(data[0]))
            else:
                polymer_coords_orig.append(np.array([]))
    except Exception as e:
        print(f"Error loading coords: {e}")
        return

    # --- 2. Recenter ---
    if recenter:
        polymer_coords = recenter_coordinates_v3(polymer_coords_orig, box_vectors)
    else:
        polymer_coords = polymer_coords_orig

    # --- 3. Color Setup ---
    n_chains = len(polymer_coords)
    chain_colors_list = []
    bead_colors_list = []
    type_legend_handles = {}

    if color_mode == "type":
        types_seq = (
            types
            if types is not None
            else getattr(traj, "chrom_seq", getattr(traj, "types", None))
        )
        if types_seq is None:
            color_mode = "chain"
        else:
            unique_types = sorted(list(set(types_seq)))
            cmap = plt.get_cmap("tab10")
            type_map = {t: cmap(i % 10) for i, t in enumerate(unique_types)}
            all_bead_colors = np.array([type_map[t] for t in types_seq])
            for sel in chain_selections:
                bead_colors_list.append(all_bead_colors[sel])
            for t, c in type_map.items():
                type_legend_handles[t] = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=c,
                    markersize=10,
                    label=f"Type {t}",
                )

    if color_mode == "chain":
        if colors is None:
            cmap = plt.get_cmap("tab10")
            chain_colors_list = [cmap(i % 10) for i in range(n_chains)]
        else:
            chain_colors_list = colors

    # --- 4. Plotting Setup ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    shifts = list(itertools.product(range(-n_layers, n_layers + 1), repeat=3))
    vec_a, vec_b, vec_c = box_vectors[0], box_vectors[1], box_vectors[2]

    # Store artists to update sizes later
    central_scatters = []
    image_scatters = []

    def plot_chain(coords, c_mode, c_idx, alpha, is_central, style):
        if coords.size == 0:
            return
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

        # Color Logic
        if c_mode == "chain":
            base_color = chain_colors_list[c_idx]
            c_scatter = base_color
            c_line = base_color
        else:
            c_scatter = bead_colors_list[c_idx]
            c_line = "gray"

        # 1. Draw Lines (Bonds)
        # Central: Standard width. Images: Thin.
        lw = 2.0 if is_central else 1.0
        # Images: Faint lines. If style is 'scatter', make lines even fainter to emphasize beads.
        l_alpha = 0.7 if is_central else (alpha * 0.5 if style == "scatter" else alpha)

        if isring:
            ax.plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                np.append(zs, zs[0]),
                color=c_line,
                alpha=l_alpha,
                linewidth=lw,
            )
        else:
            ax.plot(xs, ys, zs, color=c_line, alpha=l_alpha, linewidth=lw)

        # 2. Draw Scatter (Beads)
        # Logic: Always draw for central. For images, only if style is 'scatter'.
        if is_central or style == "scatter":
            # Initial size placeholder (will be updated)
            s_init = 1
            sc_alpha = 0.8 if is_central else alpha

            kwargs = {"alpha": sc_alpha, "s": s_init}
            if c_mode == "chain":
                kwargs["color"] = c_scatter
            else:
                kwargs["c"] = c_scatter

            sc = ax.scatter(xs, ys, zs, **kwargs)

            # Categorize for later resizing
            if is_central:
                central_scatters.append(sc)
            else:
                image_scatters.append(sc)

    # --- 5. Plotting Loop ---
    print(f"Plotting {len(shifts)} lattice copies...")
    all_plotted_points = []

    for i_grid, j_grid, k_grid in shifts:
        is_central = i_grid == 0 and j_grid == 0 and k_grid == 0
        shift_vec = i_grid * vec_a + j_grid * vec_b + k_grid * vec_c

        for c_idx, chain_coords in enumerate(polymer_coords):
            shifted_chain = chain_coords + shift_vec
            if shifted_chain.size > 0:
                all_plotted_points.append(shifted_chain[::5])  # Downsample for limits

            plot_chain(
                shifted_chain,
                color_mode,
                c_idx,
                alpha=image_alpha,
                is_central=is_central,
                style=image_style,
            )

        if is_central:
            corners = _draw_generic_box(
                ax, box_vectors, color="k", alpha=0.8, linewidth=1.5
            )
            all_plotted_points.append(corners)

    # --- 6. Limits & Projection ---
    if axis_limits:
        x_min, x_max, y_min, y_max, z_min, z_max = axis_limits
    else:
        if all_plotted_points:
            all_pts = np.vstack(all_plotted_points)
            min_ext, max_ext = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
            center = (min_ext + max_ext) / 2.0
            max_span = np.max(max_ext - min_ext)
            if max_span == 0:
                max_span = 10.0
            buffer = max_span * 0.55
            x_min, x_max = center[0] - buffer, center[0] + buffer
            y_min, y_max = center[1] - buffer, center[1] + buffer
            z_min, z_max = center[2] - buffer, center[2] + buffer
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = -10, 10, -10, 10, -10, 10

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel(r"X ($\sigma$)")
    ax.set_ylabel(r"Y ($\sigma$)")
    ax.set_zlabel(r"Z ($\sigma$)")

    if r is not None:
        ax.set_proj_type("ortho")
        ax.view_init(elev=30, azim=-45)
    else:
        ax.set_proj_type("persp")
    ax.set_box_aspect([1, 1, 1])

    if color_mode == "type":
        ax.legend(handles=type_legend_handles.values(), loc="upper right")
    elif color_mode == "chain" and n_chains <= 5:
        ax.legend(loc="upper right")
    ax.set_title(f"PBC Visualization ({n_layers} layers) - {image_style}")

    # --- 7. Apply Physical Size (The Fix) ---
    if r is not None:
        fig.canvas.draw()  # Force render to get transforms
        data_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        bbox = ax.get_window_extent()

        if bbox and data_range > 0:
            points_per_unit = (bbox.width * 72 / fig.get_dpi()) / data_range
            # Calculate physical area (s is area in points^2)
            s_phys = np.pi * ((r * points_per_unit) ** 2)
            s_phys = np.clip(s_phys, 0.01, 50000)

            # Apply to Central (Full Size)
            for sc in central_scatters:
                sc.set_sizes(np.full(len(sc.get_offsets()), s_phys))

            # Apply to Images (Scaled Down for visual clarity)
            # Factor 0.4 means radius is ~63% of central, looks good for "background"
            s_image = s_phys * 0.4
            for sc in image_scatters:
                sc.set_sizes(np.full(len(sc.get_offsets()), s_image))
    else:
        # Default size if r is not provided
        for sc in central_scatters:
            sc.set_sizes([20])
        for sc in image_scatters:
            sc.set_sizes([10])

    # --- 8. Output ---
    if output_name:
        plt.savefig(output_name, dpi=300)
        print(f"Plot saved to {output_name}")
        plt.close(fig)
    else:
        plt.show()
