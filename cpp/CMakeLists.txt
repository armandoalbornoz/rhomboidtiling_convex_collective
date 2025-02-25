# CMake for order-k Delaunay triangulations using CGAL.
# Includes a commandline tool (orderk) and tests (test).

cmake_minimum_required(VERSION 3.10)  # Updated to a more modern CMake version
project(orderk)

# Set C++ standard to 14 (required by CGAL 5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CGAL and its dependencies
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

# Find Boost (required by CGAL)
find_package(Boost REQUIRED)

# Find GMP and MPFR (required by CGAL)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# Add executable for the main program
add_executable(orderk src/main.cpp)

# Link CGAL and its dependencies to the main executable
target_link_libraries(orderk
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
)

# Add executable for the tests
add_executable(tests src/tests.cpp)

# Link CGAL and its dependencies to the tests executable
target_link_libraries(tests
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
)

# Add the executables to CGAL's list of targets
add_to_cached_list(CGAL_EXECUTABLE_TARGETS orderk tests)