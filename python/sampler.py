import numpy as np
import matplotlib.pyplot as plt


def generate_2d_gaussian_points(
        num_points: int,
        mean: list = [0.0, 0.0],
        cov: list | np.ndarray = [[1.0, 0.0], [0.0, 1.0]],
        seed: int = None,
        visualize: bool = True
) -> np.ndarray:
    """
    Generates 2D points from a Gaussian distribution and optionally plots them.

    Parameters:
        num_points: Number of points to generate
        mean: List/tuple of length 2 specifying the mean (default: [0, 0])
        cov: 2x2 covariance matrix (default: identity matrix)
        seed: Random seed for reproducibility
        visualize: Whether to show a scatter plot

    Returns:
        points: Array of shape (num_points, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate inputs
    if len(mean) != 2:
        raise ValueError("Mean must be a list/tuple of length 2")
    if np.array(cov).shape != (2, 2):
        raise ValueError("Covariance must be a 2x2 matrix")

    # Generate points
    points = np.random.multivariate_normal(mean, cov, num_points)

    # Visualization
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], alpha=0.6,
                    edgecolor='white', s=50, color='blue')
        plt.title(f"{num_points} Points from 2D Gaussian Distribution\n"
                  f"μ = {mean}, Σ = {np.array(cov).tolist()}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#f0f0f0')
        plt.show()

    return points