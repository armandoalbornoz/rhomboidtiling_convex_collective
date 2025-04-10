import numpy as np
import matplotlib.pyplot as plt
from orderk_delaunay import OrderKDelaunay
from matplotlib.animation import FuncAnimation
from rhomboidtiling_convex_collective.python.plotter import Plotter2D
from scipy.spatial import ConvexHull


def compute_triangle_circumcenter(A,B,C):
    """
    Computes the circumcenter of a triangle ABC.
    :param A: vertex
    :param B: vertex
    :param C: vertex
    :return:  Returns the centroid if points are colinear (degenerate triangle).
    """
    ax, ay = A
    bx, by = B
    cx, cy = C

    # Compute denominator
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        return (A + B + C) / 3  # Fallback to centroid

    # Compute circumcenter coordinates
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return np.array([ux, uy])


def get_triangle_angles(vertex1, vertex2, vertex3):
    """
      Computes the angles of a triangle given its three vertices using the law of cosines.

      Returns:
          angles: A list of angles (in radians).
      """
    a = np.linalg.norm(vertex2 - vertex3)
    b = np.linalg.norm(vertex1 - vertex3)
    c = np.linalg.norm(vertex1 - vertex2)

    angle1 = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
    angle2 = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
    angle3 = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    return [angle1, angle2, angle3]

def compute_triangle_angles(points, vertices, triangles):
    """
     Computes angles for 2D triangles in order-2 Delaunay mosaic.
     Parameters:
         points: Input points (2D array).
         vertices: List of vertex tuples (k-tuples of point indices).
         simplices: List of simplices (triangles) as indices into `vertices`.
     Returns:
         List of dictionaries with keys "simplex" and "angles_deg".
     """


    triangle_angles = []

    for triangle in triangles:

        # Get the tuples of original points for each vertex in the triangle
        vertex_tuples = [vertices[i] for i in triangle]

        #Compute the actual geometric vertices using average
        average1 = np.mean(points[list(vertex_tuples[0])], axis=0)
        average2 = np.mean(points[list(vertex_tuples[1])], axis=0)
        average3 = np.mean(points[list(vertex_tuples[2])], axis=0)

        # Compute the angles of the current triangle
        angles_rad = get_triangle_angles(average1, average2, average3)
        angles_deg = np.degrees(angles_rad)

        # Store angles with its corresponding triangle

        triangle_angles.append({"triangle": triangle, "angles_deg": angles_deg })

    return triangle_angles


def ruppert_refinement(
        initial_points: np.ndarray,
        order: int = 1,
        threshold: float = 15.0,
        draw_mosaic: bool = True,
        show_angle_histogram: bool = True,
        max_iterations: int = 5,
        verbose: bool = True
) -> np.ndarray:
    """
    Refines Delaunay Triangulation/Mosaic incrementally using Ruppert's algorithm.
    """
    points = initial_points.copy()
    iteration = 0
    converged = False

    while iteration < max_iterations and not converged:
        # Compute current Delaunay triangulation/mosaic
        orderk_delaunay = OrderKDelaunay(points, order=order)
        vertices = orderk_delaunay.diagrams_vertices[order - 1]
        simplices = orderk_delaunay.diagrams_simplices[order - 1]
        triangle_angles = compute_triangle_angles(points, vertices, simplices)

        # Visualize current state
        if draw_mosaic and iteration % 10 == 0:
            plotter = Plotter2D(points, orderk_delaunay)
            plotter.draw(order)
            plot_angle_histogram(triangle_angles)

        # Find all bad triangles
        bad_triangles = [entry for entry in triangle_angles
                         if np.min(entry["angles_deg"]) < threshold]

        if not bad_triangles:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break

        # Select the worst triangle (smallest minimum angle)
        worst_triangle = min(bad_triangles, key=lambda x: np.min(x["angles_deg"]))

        # Get geometric vertices and compute circumcenter
        vertex_tuples = [vertices[i] for i in worst_triangle["triangle"]]
        avg_vertices = [np.mean(points[list(vt)], axis=0) for vt in vertex_tuples]
        circumcenter = compute_triangle_circumcenter(*avg_vertices)

        # Add new point and increment iteration
        points = np.vstack([points, circumcenter])
        iteration += 1

        if verbose:
            print(f"Iteration {iteration}: Added 1 new point")

    # Final angle histogram
    # if show_angle_histogram:
    #     orderk_delaunay = OrderKDelaunay(points, order=order)
    #     vertices = orderk_delaunay.diagrams_vertices[order - 1]
    #     simplices = orderk_delaunay.diagrams_simplices[order - 1]
    #     triangle_angles = compute_triangle_angles(points, vertices, simplices)
    #     plot_angle_histogram(triangle_angles)

    return points

def animate_refinement(initial_points, order=1, threshold=15.0, max_iterations=5, interval=500):
    """
    Animates the refinement process with real-time updates of triangulation and angle histogram.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    points = initial_points.copy()
    all_angles = []

    def update(frame):
        nonlocal points
        ax1.clear()
        ax2.clear()

        # Compute current triangulation
        orderk_delaunay = OrderKDelaunay(points, order=order)
        vertices = orderk_delaunay.diagrams_vertices[order - 1]
        simplices = orderk_delaunay.diagrams_simplices[order - 1]

        # Update triangulation plot
        plotter = Plotter2D(points, orderk_delaunay)
        plotter.draw(order, ax=ax1)
        ax1.set_title(f"Iteration {frame + 1}\nPoints: {len(points)}")

        # Compute and store angles
        triangle_angles = compute_triangle_angles(points, vertices, simplices)
        current_angles = [a for entry in triangle_angles for a in entry["angles_deg"]]
        all_angles.extend(current_angles)

        # Update histogram
        ax2.hist(all_angles, bins=np.linspace(0, 180, 37), edgecolor='black')
        ax2.set_xlim(0, 180)
        ax2.set_xticks(np.arange(0, 181, 30))
        ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Angle Distribution (Min: {np.min(current_angles):.1f}°)")

        # Find and plot circumcenters
        new_points = []
        for entry in triangle_angles:
            if np.min(entry["angles_deg"]) < threshold:
                simplex = entry["triangle"]
                vertex_tuples = [vertices[i] for i in simplex]
                avg1 = np.mean(points[list(vertex_tuples[0])], axis=0)
                avg2 = np.mean(points[list(vertex_tuples[1])], axis=0)
                avg3 = np.mean(points[list(vertex_tuples[2])], axis=0)
                circumcenter = compute_triangle_circumcenter(avg1, avg2, avg3)
                new_points.append(circumcenter)
                ax1.plot(circumcenter[0], circumcenter[1], 'rx', markersize=8)

        # Add new points for next iteration
        if new_points:
            points = np.vstack([points, new_points])

        return ax1, ax2

    ani = FuncAnimation(fig, update, frames=max_iterations, interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani

def plot_angle_histogram(triangle_angles, bins=50, color='skyblue'):
    """
    Plots a histogram of triangle angles (in degrees).

    Parameters:
        triangle_angles: Output from compute_triangle_angles.
        bins: Number of histogram bins (default: 18 for 10-degree increments).
        color: Color of the histogram bars.
    """
    # Extract all angles into a single list
    all_angles = []
    for entry in triangle_angles:
        all_angles.extend(entry["angles_deg"])

    # Plot histogram
    plt.figure(figsize=(15, 6))
    plt.hist(all_angles, bins=bins, edgecolor='black', color=color)

    # Add labels and title
    plt.xlabel("Angle (degrees)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Triangle Angles in Delaunay Mosaic", fontsize=14)
    plt.grid(axis='y', alpha=0.75)

    # Add vertical line at 60° for reference (equilateral triangles)
    plt.axvline(60, color='red', linestyle='--', label='Equilateral Angle (60°)')
    plt.legend()

    # Set x-axis limits and custom ticks for better separation
    plt.xlim(0, 180)
    plt.xticks(np.arange(0, 181, 5))

    plt.show()

