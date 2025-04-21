
import matplotlib
matplotlib.use('Agg')  # non-interactive, no Qt required

import io
import base64
import triangle as tr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from rhomboidtiling_convex_collective.refinementlib.angles import (
    compute_triangle_angles,
    refinement_histogram
)
from rhomboidtiling_convex_collective.refinementlib.orderk_delaunay import OrderKDelaunay
from rhomboidtiling_convex_collective.refinementlib.plotter import Plotter2D

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

def refine_with_k_steiner_points(points, k=5, min_angle=20):
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
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.5, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_hist1 = fig.add_subplot(gs[1, 0])
    ax_hist2 = fig.add_subplot(gs[1, 1])

    def update(frame):
        ax1.cla(); ax2.cla(); ax_hist1.cla(); ax_hist2.cla()
        mesh = refinement_steps[frame]
        pts = mesh["vertices"]
        steiner_count = len(pts) - len(original_points)

        order1 = OrderKDelaunay(pts, order=1)
        Plotter2D(pts, order1).draw(order=1, ax=ax1)
        ax1.set_title(f"Order-1 | Steiner Points: {steiner_count}")
        angles1 = compute_triangle_angles(pts, order1.diagrams_vertices[0], order1.diagrams_simplices[0])
        refinement_histogram(angles1, ax_hist1)

        order2 = OrderKDelaunay(pts, order=2)
        Plotter2D(pts, order2).draw(order=2, ax=ax2)
        ax2.set_title(f"Order-2 | Steiner Points: {steiner_count}")
        angles2 = compute_triangle_angles(pts, order2.diagrams_vertices[1], order2.diagrams_simplices[1])
        refinement_histogram(angles2, ax_hist2)

        return ax1, ax2, ax_hist1, ax_hist2

    return FuncAnimation(fig, update, frames=len(refinement_steps), interval=500, repeat=False)


def get_refinement_frames(refinement_steps, original_points, min_angle):
    frames = []
    for mesh in refinement_steps:
        fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.5, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_hist1 = fig.add_subplot(gs[1, 0])
        ax_hist2 = fig.add_subplot(gs[1, 1])

        pts = mesh["vertices"]
        steiner_count = len(pts) - len(original_points)

        order1frame = OrderKDelaunay(pts, order=1)
        Plotter2D(pts, order1frame).draw(order=1, ax=ax1)
        ax1.set_title(f"Order-1 | Steiner: {steiner_count}")
        angles1 = compute_triangle_angles(pts, order1frame.diagrams_vertices[0], order1frame.diagrams_simplices[0])
        refinement_histogram(angles1, ax_hist1)

        # order‑2
        order2frame = OrderKDelaunay(pts, order=2)
        Plotter2D(pts, order2frame).draw(order=2, ax=ax2)
        ax2.set_title(f"Order-2 | Steiner: {steiner_count}")
        angles2 = compute_triangle_angles(pts, order2frame.diagrams_vertices[1], order2frame.diagrams_simplices[1])
        refinement_histogram(angles2, ax_hist2)

        buf = io.BytesIO() # png of figure to mem
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # base64 encoding to array
        data_uri = (
          "data:image/png;base64,"
          + base64.b64encode(buf.read()).decode('ascii')
        )
        frames.append(data_uri)

    return frames # send array thru api animate endpoint, port 5000
