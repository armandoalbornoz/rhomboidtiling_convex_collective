import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from scipy.spatial import ConvexHull, Delaunay
import io
import base64

def set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim3d(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim3d(z_mid - max_range/2, z_mid + max_range/2)

def generate_convex_hull_animation(points=None, num_points=20, seed=42):
    if points is None:
        np.random.seed(seed)
        points = np.random.rand(num_points, 2)
    else:
        points = np.array(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be an (n,2) array")

    pts3 = np.column_stack([points, points[:,0]**2 + points[:,1]**2])

    idx = np.argsort(points[:,0])
    pts2_sorted = points[idx]
    pts3_sorted = pts3[idx]

    frames = []
    total = len(pts2_sorted)

    for i in range(3, total + 1):
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        current2 = pts2_sorted[:i]
        current3 = pts3_sorted[:i].copy()

        ax1.scatter(points[:,0], points[:,1], s=20, alpha=0.5)
        if i >= 3:
            tri2 = Delaunay(current2)
            for simplex in tri2.simplices:
                seg = np.vstack([current2[simplex], current2[simplex[0]]])
                ax1.plot(seg[:,0], seg[:,1], color='gray', linewidth=1, alpha=0.7)
        ax1.scatter(current2[-1,0], current2[-1,1], color='red', s=50, zorder=5)
        ax1.set_title(f'2D Delaunay: {i} pts')
        ax1.set_aspect('equal')
        ax1.grid(True, linestyle='--', alpha=0.5)

        if i >= 4:
            hull3 = ConvexHull(current3)

            for simplex in hull3.simplices:
                face = current3[simplex]
                poly = art3d.Poly3DCollection(
                    [face],
                    facecolor='lightgray',
                    edgecolor='k',
                    linewidth=0.5,
                    alpha=0.3
                )
                ax2.add_collection3d(poly)

            # 2) Overplot the “lower hull” faces in solid red
            for simplex in hull3.simplices:
                tri = current3[simplex]
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if normal[2] < 0:  # facing downward
                    red_poly = art3d.Poly3DCollection(
                        [tri],
                        facecolor='red',
                        edgecolor='darkred',
                        linewidth=1.0,
                        alpha=0.8
                    )
                    ax2.add_collection3d(red_poly)
        else:
            ax2.text(0.5, 0.5, 0.5,
                     f"Need 4 pts for 3D hull (currently {i})",
                     ha='center', va='center',
                     transform=ax2.transAxes)

        ax2.scatter(current3[:,0], current3[:,1], current3[:,2],
                    s=20, alpha=0.5)
        ax2.scatter(current3[-1,0], current3[-1,1], current3[-1,2],
                    color='red', s=50, zorder=5)

        ax2.set_title(f'3D Convex Hull: {i} pts')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=30, azim=(i*10) % 360)

        try:
            ax2.set_box_aspect([1, 1, 0.5])
        except AttributeError:
            set_axes_equal(ax2)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append('data:image/png;base64,' +
                      base64.b64encode(buf.read()).decode())

    return frames