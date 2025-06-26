import numpy as np
from utils.utils import projection_kernel
from utils.clustering import kmeans_fixed_centers
from functools import partial

from data.data_handling import DataSet

def algorithm_1(A, centers, points_M_1:DataSet, epsilon=1e-2, dist=partial(np.linalg.norm, "fro")):
    """
        Implements the iterative diameter estimation algorithm for each randomly sampled point in the input set.
        Arguments:
        - A: The matrix of the forward operatior (for which we are computing the Moore-Penrose inverse 
             with a truncated singular value decomposition).
        - centers: randomly sampled points from input set.
        - points: The points in the target set M_1.
        - epsilon: Tolerance for noisy case.

        Returns:
        - diameters: The estimated diameters for each randomly sampled point in the input set.
    """

    # Compute Moore-Penrose-Inverse of A
    A_dagger = np.linalg.pinv(A)
    A_dagger_A = np.dot(A_dagger, A)
    
    diameters = np.zeros(shape=[len(centers)])

    for c_i in centers:
        # prepare cluster center
        c_i = np.hstack((c_i, 0))  # extend to 3D for compatibility with A
        A_dagger_A_x_perp_i = np.dot(A_dagger_A, np.dot(A_dagger, c_i)) # projection onto range of A
        diam_F_c_i = 0

        # iterate over batches in point set
        for points in points_M_1:
            # check if point satisfies the condition
            condition = dist(np.dot(A_dagger_A, points)-A_dagger_A_x_perp_i, axis=[-1,-2])
            
            cond_mask = condition <= 2*epsilon
            
            if any(cond_mask):
                # print(f"Point satisfying condition found (ε = {epsilon})")
                # calculate projection onto nullspace of A
                proj_nullspace = projection_kernel(A, points[cond_mask])
                max_diam_x_n = np.max(2*dist(proj_nullspace, axis=[-1,-2]))

                # update diameter if necessary
                if max_diam_x_n > diam_F_c_i:
                    diam_F_c_i = max_diam_x_n

        # Store final diameter for this cluster center
        diameters  = np.where(diam_F_c_i)

    return diameters


def diameter_for_different_ks(A, points_M_1, points_M_2, max_k, eps=1e-1):
    """
    Returns the approximate worst-case kernel size, as maximum over diameters of feasible sets, 
    for an arbitrary number of sampled points and all diameters.
    Arguments:
        - A: the matrix of the forward operatior (for which we are computing the Moore-Penrose inverse 
            with a truncated singular value decomposition).
        - points_M_1: randomly sampled points in the target dataset.
        - points_M_2: randomly sampled points in the input dataset.
        - max_k: maximum number of points sampled for both input and target dataset.
        TODO: one max for number of points in input dataset and one max for number points in target dataset.
        TODO: badd parameter to either return all diameters or only max diameter
        - eps: precision of feasible set membership in Algorithm 1 for computing the diameters of the feasible sets.
    Returns:
        - max_diameters: diameters of feasible sets.
        - max_diameter_total: approximate worst-case kernel size, as maximum over diameters of feasible sets.
    """
    max_diameters = []
    random_points_list = kmeans_fixed_centers(points_M_2, max_k)
    for cluster_centers in random_points_list:
        diameters = algorithm_1(A, cluster_centers, points_M_1, eps)
        max_diameter = max(diameters)
        print(f"for k={len(diameters)} we get max_diameter={max_diameter}")
        max_diameters.append(max_diameter)
        max_diameter_total = max(max_diameters)

    return max_diameters, max_diameter_total 

def error_approximation_wckernelsize(max_diameter_total, ker_size):
    """ Computes error of approximation to the worst-case kernel size 
        if an analytical value for the worst-case kernel size is available.
        Arguments:
        - max_diameters: approximation to worst-case kernel size.
        - ker_size: analytical value for worst-case kernel size.
        Returns:
        - rel_error: positive relative error if approximation correctly
         approximates the worst-case kernel size from below.
    """
    rel_error = (max_diameter_total - ker_size) / ker_size
    assert rel_error >= 0, "Ups, we failed haha."
    
    return rel_error