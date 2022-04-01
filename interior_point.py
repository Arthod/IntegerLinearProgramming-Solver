import numpy as np
#import matplotlib.pyplot as plt

def interior_point_iteration(A, c, x_initial):
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
    v = abs(min(c_p))

    alpha = 0.5
    x_tilde = np.ones(len(c_p)) + (alpha / v) * c_p

    # 5
    x = D @ x_tilde
    return x

def interior_point(A, c, x_initial):
    tol = 10e-5
    path = [x_initial]

    x_prev = -10 * np.ones(c.size)
    x = np.copy(x_initial)
    while (not all(np.isclose(x_prev, x, rtol=tol))):
        x_prev = x
        x = interior_point_iteration(A, c, x)
        path.append(x)

    return x, path


def is_feasible(A, b, x):
    return all(np.isclose(A @ x, b))

if __name__ == "__main__":
    
    A = np.array([[1, 1, 1]])
    b = np.array([8])
    c = np.array([1, 2, 0])
    x_initial = np.array([2, 2, 4])


    """
    A = np.array([[1, 2, 1, 0, 0], [3, 1, 0, 1, 0], [1.5, 2, 0, 0, 1]])
    b = np.array([21, 40, 24])
    c = np.array([1, 2, 0, 0, 0])    # TODO
    x_initial = np.array([2, 2, 15, 32, 17])
    """


    """
    x_initial = np.random.dirichlet(np.ones(c.size)/np.sum(b))
    while (not is_feasible(A, b, x_initial)):
        x_initial = np.random.dirichlet(np.ones(c.size)) * np.sum(b)
        print(np.sum(x_initial))
        print(x_initial)
    """
    assert is_feasible(A, b, x_initial), "Initial solution not feasible: " + str(A@x_initial)
    print(A, b, c, x_initial)
    x, path = interior_point(A, c, x_initial)
    print("z = " + str(x @ c))
    print("x = " + str(x))

    #plot_path(path, A, b)
    
    