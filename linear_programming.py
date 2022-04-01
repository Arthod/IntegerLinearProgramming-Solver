from sympy import Symbol, Poly
import numpy as np
from scipy.optimize import linprog

from interior_point import interior_point


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
        
        A = []
        for constraint in self.constraints:
            l = []
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            for id, symbol in self.variables["x"].items():
                l.append(coeff_dict.get(symbol, 0))
            A.append(l)

        A = np.vstack(np.array(A), np.identity(len(A))) 
        print(A)
        b = np.array([constraint.expr_RHS for constraint in self.constraints])
        c = np.array([self.max_or_min * coeff for coeff in Poly(self.objective).coeffs()])
        x_initial = np.array([2, 2])

        return interior_point(A, c, x_initial)


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
