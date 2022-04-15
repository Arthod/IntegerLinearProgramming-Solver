
import numpy as np

"""
A, b, c

A = [[ 1.  1.  1.  1.  1.  0.  0.]
    [ 2.  1. -1. -1.  0. -1.  0.]
    [ 0. -1.  0.  1.  0.  0. -1.]],
b = [40 10 12],
c = [-0.5 -3.  -1.  -4.   0.   0.   0. ])
"""


def simplex(A: np.array, b: np.array, c: np.array, slacks_amt: int) -> "tuple[np.array, int]":
    # https://sdu.itslearning.com/ContentArea/ContentArea.aspx?LocationID=17727&LocationType=1&ElementID=589350
    #x_current = basic_feasible_solution(A, b, c)
    #N = [i for i in range(c.size - slacks_amt)]   # Not In basis
    B = [i + (c.size - slacks_amt) for i in range(slacks_amt)]    # In basis
    # Zero basic solution

    iterations_count = 0

    print(A, b, c)

    while True:
        iterations_count += 1
        ## Choosing a pivot
        # Pick a non-negative column    TODO
        positives = np.where(c > 0)
        if (not len(positives[0])):
            break

        xx = positives[0][0]
        yy = None
        yy_old = None

        # Picking smallest row b[i] / a[i]
        val = float("inf")
        a = A[:,xx]
        for i in range(b.size):
            if (a[i] > 0):
                val_current = b[i] / a[i]
                if (val_current <= val):
                    yy = i
                    val = val_current

        # Find column from where basis was exited
        a = A[yy,:]
        for i in range(a.size):
            if (a[i] == 1):
                s = np.sum(A[:,i]) + c[i]
                if (s == 1):
                    yy_old = i
                    break
        # Remove yy_old from basis and add yy to basis
        B.remove(yy_old)
        B.append(yy)
        #N.append(yy_old)
        #N.remove(yy)

        ## Pivot: A[xx, yy] - Peform row operations
        # Set yy row with in column xx to 1
        b[yy] = b[yy] * 1 / A[yy, xx]
        A = row_mult(A, yy, 1 / A[yy, xx])

        # Set column xx in c to 0
        c = c - A[yy,:] * c[xx]

        # Set other values in column xx to 0
        for i in range(A.shape[0]):
            if (i == yy):
                continue
            b[i] = b[i] - A[i, xx] * b[yy]
            A = row_add(A, i, yy, -A[yy, xx] * A[i, xx])

    # Get solution
    x_sol = np.zeros(c.size)
    for i in range(len(B)):
        x_sol[B[i]] = b[i]
    return np.array(x_sol), iterations_count

def row_mult(A: np.array, row_index: int, scalar: int) -> np.array:
    # Row operation: R1 = scalar * R1 (TODO - optimize)
    # Inspiration:
    # https://personal.math.ubc.ca/~pwalls/math-python/linear-algebra/solving-linear-systems/
    I = np.eye(A.shape[0])
    I[row_index, row_index] = scalar
    return I @ A

def row_add(A: np.array, row_main_index: int, row_other_index: int, scalar: int) -> np.array:
    # Row operation: R1 = R1 + scalar * R2 (TODO - optimize)
    # Inspiration:
    # https://personal.math.ubc.ca/~pwalls/math-python/linear-algebra/solving-linear-systems/
    
    I = np.eye(A.shape[0])
    if (row_main_index == row_other_index):
        I[row_main_index, row_main_index] = scalar + 1
    else:
        I[row_main_index, row_other_index] = scalar
    return I @ A

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
    
    return zero_point


def is_feasible(A: np.array, x: np.array, b: np.array, tol=10e-5) -> bool:
    return all(np.isclose(A @ x, b, atol=tol))


def tableau(A: np.array, b: np.array, c: np.array):
    print(A, b.T, c.T)

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

