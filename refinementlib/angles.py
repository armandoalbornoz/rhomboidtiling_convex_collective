import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from rhomboidtiling_convex_collective.refinementlib.orderk_delaunay import OrderKDelaunay
from matplotlib.animation import FuncAnimation
from rhomboidtiling_convex_collective.refinementlib.plotter import Plotter2D
from scipy.spatial import ConvexHull
import triangle as tr


def get_triangle_angles(vertex1, vertex2, vertex3):
    """
      Computes the angles of a triangle given its three vertices using the law of cosines.

      Returns:
          angles: A list of angles (in radians).
      """
    a = np.linalg.norm(vertex2 - vertex3)
    b = np.linalg.norm(vertex1 - vertex3)
    c = np.linalg.norm(vertex1 - vertex2)

    # Check for degenerate triangles using area (Heron's formula)
    s = (a + b + c) / 2
    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))  # Handle numerical errors

    if area < 1e-10:
        return []  # Skip degenerate triangles


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


def compute_triangle_angles_refinement(points, vertices, triangles):
    triangle_angles = []
    for triangle in triangles:
        # Convert indices to integers
        triangle = [int(i) for i in triangle]  # <-- Fix here

        vertex_tuples = [vertices[i] for i in triangle]
        average1 = np.mean(points[list(vertex_tuples[0])], axis=0)
        average2 = np.mean(points[list(vertex_tuples[1])], axis=0)
        average3 = np.mean(points[list(vertex_tuples[2])], axis=0)
        angles_rad = get_triangle_angles(average1, average2, average3)
        angles_deg = np.degrees(angles_rad)
        triangle_angles.append({"triangle": triangle, "angles_deg": angles_deg})
    return triangle_angles


def refinement_histogram(triangle_angles, ax, bins=180, color='skyblue'):
    all_angles = []
    for entry in triangle_angles:
        all_angles.extend(entry["angles_deg"])

    ax.hist(all_angles, bins=bins, edgecolor='k', color=color, alpha=0.7)
    ax.set_xlabel("Angle (degrees)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Triangle Angles", fontsize=12)
    ax.axvline(60, color='red', ls='--', lw=1, label='60°')
    ax.legend()
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 20))
    ax.grid(axis='y', alpha=0.5)


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

