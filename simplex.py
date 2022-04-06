
import numpy as np

"""
A, b, c

A = [[ 1.  1.  1.  1.  1.  0.  0.]
    [ 2.  1. -1. -1.  0. -1.  0.]
    [ 0. -1.  0.  1.  0.  0. -1.]],
b = [40 10 12],
c = [-0.5 -3.  -1.  -4.   0.   0.   0. ])
"""


def simplex(A: np.array, b: np.array, c: np.array) -> np.array:
    # https://sdu.itslearning.com/ContentArea/ContentArea.aspx?LocationID=17727&LocationType=1&ElementID=589350
    x_current = basic_feasible_solution(A, b, c)
    print(A, b, c)

    ## Choosing a pivot
    # Pick a non-negative column    TODO
    positives = np.where(c > 0)
    if (not len(positives[0])):
        return "All coefficients negative"

    x_index = positives[0][0]
    y_index = None

    val = float("inf")
    a = A[:,x_index]
    for i in range(b.size):
        if (a[i] > 0):
            val_current = b[i] / a[i]
            if (val_current <= val):
                y_index = i
                val = val_current

    print(y_index)
    print(x_index)


def basic_feasible_solution(A: np.array, b: np.array, c: np.array) -> np.array:
    zero_point = zero_point_solution(A, b, c)
    if is_feasible(A, zero_point, b):
        return zero_point
    

"""
Feasible when all b ≥ 0, since x = 0 is feasible 
"""
def zero_point_solution(A: np.array, b: np.array, c: np.array) -> np.array:
    num_decision = len([i for i in c if i != 0])
    slack_sign = [A[i][num_decision+i] for i in range(len(b))]
    slack_values = [b[i]/slack_sign[i] for i in range(b.size)]
    zero_point = np.hstack((np.zeros(num_decision), slack_values))
    print("zero_point:", zero_point)
    print("Zero point solution is feasible:", is_feasible(A, zero_point, b))
    return zero_point


def is_feasible(A: np.array, x: np.array, b: np.array, tol=10e-5) -> bool:
    return all(np.isclose(A @ x, b, atol=tol))


if __name__ == "__main__":
    A = np.array([[ 1,  1,  1,  1,  1,  0,  0],
                  [ 2,  1, -1, -1,  0, -1,  0],
                  [ 0, -1,  0,  1,  0,  0, -1]])

    b = np.array( [40, 10, 12])
    
    c = np.array( [-0.5, -3,  -1,  -4,   0,   0,   0, ])

    print("A")
    print(A)
    print("b")
    print(b)
    print("c")
    print(c)

    basic_feasible_solution(A, b, c)

