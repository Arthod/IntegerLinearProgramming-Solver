import numpy as np
#import matplotlib.pyplot as plt

def interior_point_iteration(A, c, x_initial, alpha):
    # 1
    D = np.identity(c.size) *  x_initial
    x_tilde = np.linalg.inv(D) @ x_initial  # array of ones
    
    # 2
    A_tilde = A @ D
    c_tilde = D @ c

    # 3
    P = np.identity(c.size) - A_tilde.T @ np.linalg.inv(A_tilde @ A_tilde.T) @ A_tilde
    c_p = P @ c_tilde


    # 4
    v = np.abs(np.min(c_p))

    x_tilde = np.ones(len(c_p)) + (alpha / v) * c_p

    # 5
    x = D @ x_tilde
    return x

def interior_point(A, c, x_initial, alpha=0.5, optimal_tol=10e-5):
    iterations = 0
    path = [x_initial]

    x_prev = -10 * np.ones(c.size)
    x = np.copy(x_initial)
    while (not all(np.isclose(x_prev, x, rtol=optimal_tol))):
        x_prev = x
        x = interior_point_iteration(A, c, x, alpha)
        
        path.append(x)
        iterations += 1

    return x, path, iterations

    