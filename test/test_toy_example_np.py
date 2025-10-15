import numpy as np
from playground.generator_functions import random_uni_points_in_ball
from accuracy_bounds.inverseproblems.feasible_sets import compute_feasible_set_linear_forwardmodel
from accuracy_bounds.inverseproblems.kersize_compute import worstcase_kernelsize, worstcase_kernelsize_sym, average_kernelsize, average_kernelsize_sym
from accuracy_bounds.inverseproblems.utils import apply_forwardmodel

def test_worstcase_kersize():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2
    # Kernel Size order
    p=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    # Set the range of k values
    input_target_pairs = 3000

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical wc kersize for ball around 0 with radius r+epsilon
    wc_kernel_size_analytical = 2*radius+2*epsilon

    feasible_sets_list = []
    for y in input_data[:input_target_pairs]:
        feas_set_y = compute_feasible_set_linear_forwardmodel(A, y, target_data[:input_target_pairs], p_2, epsilon)
        feasible_sets_list.append(feas_set_y)
    
    worstcase_kersize = worstcase_kernelsize(feasible_sets_list, p_1, p)
    
    error = np.abs(worstcase_kersize-wc_kernel_size_analytical)

    assert error < 0.3, f"Analytic Worstcase Kernel Error: {error}"

def test_worstcase_kersize_sym():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2
    # Kernel Size order
    p=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    # Set the range of k values
    input_target_pairs = 3000

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical wc kersize for ball around 0 with radius r+epsilon
    wc_kernel_size_analytical = 2*radius+2*epsilon

    worstcase_kersize_sym = worstcase_kernelsize_sym(A, input_data[:input_target_pairs], target_data, p_1, p_2, p, 2*epsilon)

    error = np.abs(worstcase_kersize_sym-wc_kernel_size_analytical)

    assert error < 0.3, f"Analytic Worstcase Kernel (Sym) Error: {error}"

def test_average_kersize():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2
    # Kernel Size order
    p=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    # Set the range of k values
    input_target_pairs = 3000

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical av kersize for ball around 0 with radius r+epsilon sampled with uniform distribution
    av_kernel_size_analytical = np.multiply((radius+epsilon),np.power(1/3,1/2))

    feasible_sets_list = []
    for y in input_data[:input_target_pairs]:
        feas_set_y = compute_feasible_set_linear_forwardmodel(A, y, target_data[:input_target_pairs], p_2, epsilon)
        feasible_sets_list.append(feas_set_y)

    average_kersize = average_kernelsize(feasible_sets_list, p_1, p)

    error = np.abs(average_kersize - av_kernel_size_analytical)

    assert error < 0.3, f"Analytic Average Kernel Error: {error}"

def test_average_kersize_sym():
    num_points = 3000
    radius = 2
    center = (0,0,0)
    epsilon=1e-1

    # Distance measure
    p_1=2
    p_2=2
    # Kernel Size order
    p=2

    # Toy forward operator
    A = np.diag([1, 1, 0])  # Transformation matrix

    target_data = random_uni_points_in_ball(num_points=num_points, radius=radius+epsilon, center=center, dim=3)   
    input_data = apply_forwardmodel(A, target_data)

    #analytical av kersize for ball around 0 with radius r+epsilon sampled with uniform distribution
    av_kernel_size_analytical = np.multiply((radius+epsilon),np.power(1/3,1/2))

    # Test average symmetric kernel size computations in the limit k to infty
    average_kersize_sym = average_kernelsize_sym(A, input_data, target_data, p_1, p_2, p, epsilon)

    error = np.abs(average_kersize_sym - av_kernel_size_analytical)

    assert error < 0.3, f"Analytic Average Kernel (Sym) Error: {error}"