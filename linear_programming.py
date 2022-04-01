import scipy
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
        # Variables and constraints
        self.variables = {}
        self.constraints = []

        # Problem definition
        self.max_or_min = None
        self.objective = None
        self.A = None
        self.b = None
        self.c = None

    def add_variable(self, key: any, name: str="x") -> Symbol:
        """Adds a variable to the LP

        Args:
            key (any): The key to the variable within the group
            name (str, optional): Name of the group. Defaults to "x".

        Example:
            x1 = lp.add_variable(1, name="x")
            x2 = lp.add_variable(2, name="x")

        Raises:
            Exception: If variable key is already set within the group

        Returns:
            Symbol: Sympy symbol
        """

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


    def add_constraint(self, expr_LHS, comparator: int, expr_RHS) -> "Constraint":
        """Adds a constraint to the LP

        Args:
            expr_LHS (sympy expression): Left side of the constraint. Can use variables
            comparator (int): LP.LESS_EQUAL, LP.EQUAL, LP.GREATER_EQUAL
            expr_RHS (sympy expression): Right side of the constraint. Can use variables (not yet)

        Example:
            lp.add_constraint(2 * x2, LP.LESS_EQUAL, 14)
            lp.add_constraint(x1 + 2 * x2, LP.LESS_EQUAL, 25)
            lp.add_constraint(2 * x1 + 9 * x2, LP.LESS_EQUAL, 80)

        Returns:
            Constraint: Constraint type
        """
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

        x_initial = self.get_initial_feasible_solution()

        self.x, self.path = interior_point.interior_point(self.A, self.c, x_initial)

        return self.x

    def is_feasible(self, x):
        return all(np.isclose(self.A @ x, self.b, 0.01))

    def get_initial_feasible_solution(self):
        amount_vars_slack = self.b.size
        amount_vars = self.c.size - self.b.size

        x1 = np.hstack((np.full(amount_vars, 2), np.zeros(amount_vars_slack)))
        x = np.hstack((np.full(amount_vars, 2), self.b - (self.A @ x1)))
        
        return x

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
            # n = x when y = 0  x = c / b
            xs = np.array([0, min(3000, self.b[i] / self.A[i][0])])

            # c = ax + by   =>    y = (c - bx) / a
            ys = (self.b[i] - self.A[i][0] * xs) / self.A[i][1]
            
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

