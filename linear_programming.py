from sympy import Symbol, Poly
import numpy as np
from scipy.optimize import linprog


EQUAL = 0
GREATER_EQUAL = 1
LESS_EQUAL = 2
COMPARATORS = {
    EQUAL: "==",
    GREATER_EQUAL: ">=",
    LESS_EQUAL: "<="
    }

MAX = -1
MIN = 1


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
        return Simplex.from_lp(self).solve()


class Constraint:
    def __init__(self, expr_LHS, comparator, expr_RHS):
        self.expr_LHS = expr_LHS
        self.comparator = comparator
        self.expr_RHS = expr_RHS

    def __repr__(self):
        return str(self.expr_LHS) + COMPARATORS[self.comparator] + str(self.expr_RHS)


class Simplex:
    def __init__(self):
        """
        Maximize          z = 10*x1 + 15*x2 + 25*x3
        Subject to:       x1 + x2 + x3 <= 1000
                          x1 - 2*x2    <= 0
                                    x3 <= 340
        with              x1 >= 0, x2 >= 0
        
        A = np.array([[1, 1, 1], [1, -2, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        b = np.array([1000, 0, 340, 0, 0])
        c = np.array([10, 15, 25])
        """


        self.A = None
        self.b = None
        self.c = None


    @staticmethod
    def from_lp(lp: LP):

        A = []
        for constraint in lp.constraints:
            l = []
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            for id, symbol in lp.variables["x"].items():
                l.append(coeff_dict.get(symbol, 0))
            A.append(l)

        A = np.array(A)
        b = np.array([constraint.expr_RHS for constraint in lp.constraints])
        c = np.array([lp.max_or_min * coeff for coeff in Poly(lp.objective).coeffs()])

        simplex = Simplex()
        simplex.A = A
        simplex.b = b
        simplex.c = c

        return simplex


    def solve(self):
        res = linprog(c=self.c, A_ub=self.A, b_ub=self.b, method="revised simplex")
        print("z = ", -res.fun)
        print("x = ", res.x)

    def __repr__(self):
        # Implement tableau
        pass
