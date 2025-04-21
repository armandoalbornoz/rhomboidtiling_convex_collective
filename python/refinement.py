import matplotlib.pyplot as plt
import triangle as tr
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib.animation import FuncAnimation

from rhomboidtiling_convex_collective.python.angles import compute_triangle_angles, plot_angle_histogram, \
    refinement_histogram, compute_triangle_angles_refinement
from rhomboidtiling_convex_collective.python.orderk_delaunay import OrderKDelaunay
from rhomboidtiling_convex_collective.python.plotter import Plotter2D


# def refine_with_constraints(points, min_angle=20, max_steiner_points=1):
#     """
#       Refine a Delaunay triangulation with constraints on:
#       - Minimum angle (q)
#       - Maximum Steiner points (S)
#       """
#
#     #Compute initial Delaunay triangulation with convex hull segments ('c')
#     mesh = tr.triangulate({"vertices": points}, opts="c")
#
#     #Refine the mesh, allowing only 1 Steiner point
#     # Use 'r' (refine), 'q' (quality mesh), and 'S1' (max 1 Steiner point)
#     opts = f"rq{min_angle}S{max_steiner_points}"
#     refined_mesh = tr.triangulate(mesh, opts=opts)
#
#     # Extract vertices before and after refinement
#     original_vertices = mesh["vertices"]
#     refined_vertices = refined_mesh["vertices"]
#
#     # Identify the new Steiner point(s)
#     original_vertices = set(map(tuple, points))
#     all_vertices = refined_mesh["vertices"]
#     new_vertices = [v for v in all_vertices if tuple(v) not in original_vertices]
#
#     return refined_vertices, new_vertices


def refine_with_k_steiner_points(points,k=5, min_angle=20):
    """Refine mesh with incremental Steiner points and minimum angle constraint"""

    mesh = tr.triangulate({"vertices": points}, opts="c")  #Initial triangulation
    #original_vertices = set(map(tuple, points))
    steps = [mesh]
    steiner_added = 0

    for _ in range(k):
        prev_vertices = set(map(tuple, mesh["vertices"]))

        # Refine with  min angle and 1 Steiner point
        opts = f"rq{min_angle}S1"
        refined_mesh = tr.triangulate(mesh,opts=opts)

        # Check if new vertices (Steiner points) were added
        new_vertices = set(map(tuple, refined_mesh["vertices"])) - prev_vertices
        if len(new_vertices) == 0:
            break  # No more refinement possible

        steps.append(refined_mesh)
        mesh = refined_mesh
        steiner_added += len(new_vertices)

    return steps


def animate_refinement(refinement_steps, original_points, min_angle):
    # Initialize figure with 2 rows (mosaics) and 2 columns (histograms)
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.5, wspace=0.3)

    # Subplots for mosaics (top row)
    ax1 = fig.add_subplot(gs[0, 0])  # Order-1 mosaic
    ax2 = fig.add_subplot(gs[0, 1])  # Order-2 mosaic

    # Subplots for histograms (bottom row)
    ax_hist1 = fig.add_subplot(gs[1, 0])  # Histogram for order-1
    ax_hist2 = fig.add_subplot(gs[1, 1])  # Histogram for order-2

    def update(frame):
        ax1.cla()
        ax2.cla()
        ax_hist1.cla()
        ax_hist2.cla()

        current_mesh = refinement_steps[frame]
        refined_points = current_mesh["vertices"]
        steiner_count = len(refined_points) - len(original_points)

        # Plot order-1 Delaunay mosaic
        order1 = OrderKDelaunay(refined_points, order=1)
        plotter1 = Plotter2D(refined_points, order1)
        plotter1.draw(order=1, ax=ax1)
        ax1.set_title(f"Order-1 | Steiner Points: {steiner_count}")

        # Compute angles for order-1 (Delaunay triangulation)
        order1_vertices = order1.diagrams_vertices[0]
        order1_triangles = order1.diagrams_simplices[0]
        order1_angles = compute_triangle_angles(refined_points, order1_vertices, order1_triangles)
        refinement_histogram(order1_angles, ax_hist1)

        # Plot order-2 Delaunay mosaic
        order2 = OrderKDelaunay(refined_points, order=2)
        plotter2 = Plotter2D(refined_points, order2)
        plotter2.draw(order=2, ax=ax2)
        ax2.set_title(f"Order-2 | Steiner Points: {steiner_count}")

        # Compute angles for order-2 (mosaic cells)
        order2_vertices = order2.diagrams_vertices[1]
        order2_triangles = order2.diagrams_simplices[1]
        order2_angles = compute_triangle_angles(refined_points, order2_vertices, order2_triangles)
        refinement_histogram(order2_angles, ax_hist2)

        return ax1, ax2, ax_hist1, ax_hist2

    ani = FuncAnimation(fig, update, frames=len(refinement_steps), interval=500, repeat=False)
    plt.show()
