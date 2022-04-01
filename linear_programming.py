from sympy import Symbol, Poly
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

import interior_point


EQUAL = 0
GREATER_EQUAL = 1
LESS_EQUAL = 2
COMPARATORS = {
    EQUAL: "==",
    GREATER_EQUAL: ">=",
    LESS_EQUAL: "<="
    }

MAX = 1
MIN = -1


class LP:
    def __init__(self):
        self.variables = {}
        self.constraints = []

        self.max_or_min = None
        self.objective = None

    def add_variable(self, key, name: str="x") -> Symbol:
        # Add group of variable if not already registered
        if (name not in self.variables):
            self.variables[name] = {}

        # If key already in the group of the variable we raise an exception
        if (key in self.variables[name]):
            raise Exception("variable key already used")

        # Otherwise add the key to the variable group
        symbol = Symbol("x_" + str(key))
        self.variables[name][key] = symbol

        return symbol


    def add_constraint(self, expr_LHS: str, comparator: int, expr_RHS):
        assert comparator in COMPARATORS

        constr = Constraint(expr_LHS, comparator, expr_RHS)

        self.constraints.append(constr)

        return constr

    def set_objective(self, max_or_min, expr):
        self.max_or_min = max_or_min
        self.objective = expr

    def solve(self):
        self.b = np.array([constraint.expr_RHS for constraint in self.constraints])
        self.c = np.hstack((np.array([self.max_or_min * coeff for coeff in Poly(self.objective).coeffs()]), np.zeros(self.b.size))).astype(np.float)
        
        self.A = []
        for constraint in self.constraints:
            l = []
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            for id, symbol in self.variables["x"].items():
                l.append(coeff_dict.get(symbol, 0))
            self.A.append(l)
        self.A = np.hstack((np.array(self.A), np.identity(self.b.size))).astype(np.float)

        x_initial = np.array([2, 2, 4])

        self.x, self.path = interior_point.interior_point(self.A, self.c, x_initial)

        return self.x

    def plot_solution_path(self):
        # Plot path
        xs = []
        ys = []
        
        for p in self.path:
            xs.append(p[0])
            ys.append(p[1])
        plt.plot(xs, ys, label="path", marker="o")

        # Plot y and x axis
        plt.axhline(y=0, color="black")
        plt.axvline(x=0, color="black")

        # Plot constraints
        for i in range(self.b.size):
            xs = np.array([0, self.b[i] / self.A[i][0]])   # n = x when y = 0  x = c / b
            ys = (self.b[i] - self.A[i][0] * xs) / self.A[i][1]    # c = ax + by   =>    y = (c - bx) / a
            plt.plot(xs, ys)

        plt.legend()
        plt.show()


class Constraint:
    def __init__(self, expr_LHS, comparator, expr_RHS):
        self.expr_LHS = expr_LHS
        self.comparator = comparator
        self.expr_RHS = expr_RHS

    def __repr__(self):
        return str(self.expr_LHS) + COMPARATORS[self.comparator] + str(self.expr_RHS)


    def __repr__(self):
        # Implement tableau
        pass

