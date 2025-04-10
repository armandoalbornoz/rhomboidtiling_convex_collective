import numpy as np
from orderk_delaunay import OrderKDelaunay
from plotter import Plotter, Plotter2D, Plotter3D
from angles import compute_triangle_angles, plot_angle_histogram, animate_refinement, ruppert_refinement
from sampler import generate_2d_gaussian_points

import incremental

if __name__ == "__main__":
    # # 2D example point set
    points = np.array([[156.006,705.854], [215.257,732.63], [283.108,707.272], [244.042,670.948], [366.035,687.396], [331.768,625.715], [337.936,559.92], [249.525,582.537], [187.638,556.13], [165.912,631.197]])
   # points = generate_2d_gaussian_points(10)

    # incremental.visualize_mosaic(points, pause_time=0.0011)

    # 3D example point set
    # points = np.array([(0,0,0), (0,4,4), (4,4,0), (4,0,4), (-10,2,2)])
    # points = np.array([
    #     [0, 0],
    #     [1, 0],
    #     [0, 1],
    #     [1, 1],
    #     [0.5, 0.2],
    #     [0.2, 0.8]
    # ])

    # the order k up to which to compute the order-k Delaunay diagram

    #order = 2

    ruppert_refinement(initial_points=points, max_iterations=80)
    #animate_refinement(points, order=1, threshold=20, max_iterations=5, interval=1000)

    # Whether to print the cells of all the complexes

    #print_output = False
    # Whether to draw all the order-k Delaunay mosaics
    #draw_output = True

    # Compute the order-k Delaunay mosaics
    #orderk_delaunay = OrderKDelaunay(points, order)

    #diagrams_vertices = orderk_delaunay.diagrams_vertices[1]
    #diagrams_simplices = orderk_delaunay.diagrams_simplices[1]
    #diagram_angles = compute_triangle_angles(points,diagrams_vertices, diagrams_simplices)

    #print(f"Vertices: {diagrams_vertices}\nSimplices: {diagrams_simplices}\nAngles: {diagram_angles} " )
    #plot_angle_histogram(diagram_angles, bins=100)

    # Initialize appropriate plotter for drawing the mosaics.
    # dimension = len(points[0])
    # if dimension == 2:
    #     plotter = Plotter2D(points, orderk_delaunay)
    # elif dimension == 3:
    #     plotter = Plotter3D(points, orderk_delaunay)
    # else:
    #     # Stub that doesn't draw anything.
    #     plotter = Plotter(points, orderk_delaunay)
    #
    # for k in range(1, order+1):
    #     if draw_output:
    #         plotter.draw(k)
    #
    #     if print_output:
    #         cells = orderk_delaunay.diagrams_cells[k-1]
    #         # Output all the cells.
    #         print("Order {}. Number of cells: {}".format(
    #               len(cells[0][0]), len(cells)))
    #         for cell in sorted(cells):
    #             print(cell)
