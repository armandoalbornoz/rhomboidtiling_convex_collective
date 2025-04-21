import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def lift_point(pt):
    return np.array([pt[0], pt[1], pt[0] ** 2 + pt[1] ** 2])


class IncrementalOrder2Delaunay:
    def __init__(self):
        self.points_2d = []
        self.barycenters_3d = []
        self.hull = None

    def add_point(self, new_pt):
        # Generate new barycenters with all existing points
        new_barys = []
        for pt in self.points_2d:
            bary = (lift_point(pt) + lift_point(new_pt)) / 2
            new_barys.append(bary)
        self.points_2d.append(new_pt)
        self.barycenters_3d.extend(new_barys)

        # Initialize or update the convex hull incrementally with 'QJ' to joggle points
        if self.hull is None:
            if len(self.barycenters_3d) >= 4:  # Minimum 4 points in 3D
                try:
                    self.hull = ConvexHull(
                        np.array(self.barycenters_3d),
                        incremental=True,
                        qhull_options="QJ"  # Joggle points to avoid problems with coplanarity
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize hull: {e}")
        else:
            if new_barys:
                self.hull.add_points(np.array(new_barys))

    def get_lower_facets(self):
        if self.hull is None:
            return []
        lower_facets = []
        for eq, simplex in zip(self.hull.equations, self.hull.simplices):
            if eq[2] < 0:  # Normal points downward
                facet_points = self.hull.points[simplex]
                lower_facets.append(facet_points)
        return lower_facets

    def project_to_2d(self, faces):
        return [[p[:2] for p in face] for face in faces]


def visualize_mosaic(points, pause_time=0.5):
    """
    Visualize the incremental construction of the order-2 Delaunay mosaic
    for an arbitrary set of 2D points.

    Parameters:
    points (list or np.array): List of 2D points in format [[x1,y1], [x2,y2], ...]
    pause_time (float): Time to pause between point additions (in seconds)
    """
    delaunay = IncrementalOrder2Delaunay()
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Convert to numpy array and check dimensions
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input must be a 2D array of shape (n, 2)")

    # Calculate axis limits
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2

    for idx, pt in enumerate(points):
        delaunay.add_point(pt)
        lower_facets = delaunay.get_lower_facets()
        projected = delaunay.project_to_2d(lower_facets)

        ax.clear()
        # Plot input points
        ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.3)
        ax.scatter([p[0] for p in delaunay.points_2d],
                   [p[1] for p in delaunay.points_2d],
                   color='blue', label='Added Points')

        # Plot order-2 mosaic
        for face in projected:
            if len(face) >= 3:  # Need at least 3 points for a polygon
                poly = Polygon(face, closed=True, edgecolor='green',
                               fill=False, linewidth=1.5)
                ax.add_patch(poly)

        # Set axis limits
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(f"Order-2 Delaunay Mosaic ({idx + 1}/{len(points)} points added)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.pause(pause_time)

    plt.show()


