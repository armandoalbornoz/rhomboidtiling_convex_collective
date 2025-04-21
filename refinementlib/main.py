import numpy as np
from orderk_delaunay import OrderKDelaunay
from plotter import Plotter, Plotter2D, Plotter3D
from angles import compute_triangle_angles, plot_angle_histogram
from rhomboidtiling_convex_collective.python.refinement import animate_refinement, refine_with_k_steiner_points
from sampler import generate_2d_gaussian_points
import incremental
import triangle as tr
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # # 2D example point set
    points = np.array([[156.006,705.854], [215.257,732.63], [283.108,707.272], [244.042,670.948], [366.035,687.396], [331.768,625.715], [337.936,559.92], [249.525,582.537], [187.638,556.13], [165.912,631.197]])
    #points = generate_2d_gaussian_points(50)

    # incremental.visualize_mosaic(points, pause_time=0.0011)

    # 3D example point set

    #refined_points, new_vertices = refine_with_constraints(points, max_steiner_points=30)

    # Usage
    min_angle = 30  # User-defined parameter
    refinement_steps = refine_with_k_steiner_points(points, k=20, min_angle=min_angle)
    animate_refinement(refinement_steps, points, min_angle)  # Pass min_angle here

    # the order k up to which to compute the order-k Delaunay diagram

    order = 1

    # Whether to print the cells of all the complexes

    print_output = False
    # Whether to draw all the order-k Delaunay mosaics
    draw_output = True

    # Compute the order-k Delaunay mosaics
    # orderk_delaunay = OrderKDelaunay(points, order)
    # orderk_delaunay_refined = OrderKDelaunay(refined_points, order)

#
    # diagrams_vertices = orderk_delaunay.diagrams_vertices[1]
    # diagrams_simplices = orderk_delaunay.diagrams_simplices[1]
    # diagram_angles = compute_triangle_angles(points,diagrams_vertices, diagrams_simplices)

    #print(f"Vertices: {diagrams_vertices}\nSimplices: {diagrams_simplices}\nAngles: {diagram_angles} " )
    #plot_angle_histogram(diagram_angles, bins=100)

   # Initialize appropriate plotter for drawing the mosaics.
   #  plotter = Plotter2D(points, orderk_delaunay)
   #  plotter_refined = Plotter2D(refined_points, orderk_delaunay_refined)

    #
    # for k in range(1, order+1):
    #     if draw_output:
    #         plotter.draw(k)
    #         plotter_refined.draw(k)

        # if print_output:
        #     cells = orderk_delaunay_refined.diagrams_cells[k-1]
        #     # Output all the cells.
        #     print("Order {}. Number of cells: {}".format(
        #           len(cells[0][0]), len(cells)))
        #     for cell in sorted(cells):
        #         print(cell)
