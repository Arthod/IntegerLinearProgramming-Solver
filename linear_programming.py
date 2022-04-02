import itertools
import scipy
from sympy import Symbol, Poly
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

import interior_point


GREATER_EQUAL = -1
EQUAL = 0
LESS_EQUAL = 1
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
        self.constraints: list["Constraint"] = []

        # Problem definition
        self.is_maximizing = None
        self.objective = None
        self.A = None
        self.b = None
        self.c = None
        self.iterations = 0


    def compute_feasible_basis(self):
        # LP for finding feasible basis
        F_LP = LP()
        
        # Get variables from original problem
        F_LP.variables = self.variables.copy()

        # Get contraints from original problem
        F_LP.constraints = self.constraints.copy()
        
        # Add SLACK and FEAS variables, and add them to constraints
        
        for i in range(len(F_LP.constraints)):
            slack = F_LP.add_variable(i, name="SLACK")
            feas  = F_LP.add_variable(i, name="FEAS")
            if F_LP.constraints[i].comparator != EQUAL:
                F_LP.constraints[i].comparator = EQUAL
                F_LP.constraints[i].expr_LHS += slack
            F_LP.constraints[i].expr_LHS += feas
            if F_LP.constraints[i].expr_RHS < 0:
                F_LP.constraints[i].expr_LHS *= -1
                F_LP.constraints[i].expr_RHS *= -1
        
        
        # Objective function of feasibility LP
        F_LP.set_objective(MAX, -sum(F_LP.variables["FEAS"].values())*self.is_maximizing)

        # Solve feasibility LP and remove introduces variables
        feasible_solition = F_LP.solve(first=True)
        feasible_solution.pop("FEAS")

        # Return if feasible
        assert self.is_feasible(self, feasible_solution), "Didn't find feasible basis"
        return feasible_solution


    def add_variable(self, key: any, name: str="x", lb=None) -> Symbol:
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
        symbol = Symbol(name + "_" + str(key))
        self.variables[name][key] = symbol

        # Add a constraint for lower bound
        if (lb != None):
            self.add_constraint(symbol, GREATER_EQUAL, lb)

        return symbol


    def add_constraint(self, expr_LHS, comparator: int, expr_RHS) -> None:
        """Adds a constraint to the LP

        Args:
            expr_LHS (sympy expression): Left side of the constraint. Can use variables
            comparator (int): LP.LESS_EQUAL, LP.EQUAL, LP.GREATER_EQUAL
            expr_RHS (sympy expression): Right side of the constraint. Can use variables (not yet)

        Example:
            lp.add_constraint(2 * x2, LP.LESS_EQUAL, 14)
            lp.add_constraint(x1 + 2 * x2, LP.LESS_EQUAL, 25)
            lp.add_constraint(2 * x1 + 9 * x2, LP.LESS_EQUAL, 80)
        """
        assert comparator in COMPARATORS

        if (comparator == EQUAL):
            self.constraints.append(Constraint(expr_LHS, GREATER_EQUAL, expr_RHS))
            self.constraints.append(Constraint(expr_LHS, LESS_EQUAL, expr_RHS))
        else:
            self.constraints.append(Constraint(expr_LHS, comparator, expr_RHS))

    def set_objective(self, is_maximizing, expr):
        self.is_maximizing = is_maximizing
        self.objective = expr

    def solve(self, A_provided: bool=False, first: bool=False):
        self.b = np.array([constraint.expr_RHS for constraint in self.constraints])
        self.c = np.hstack((np.array([self.is_maximizing * coeff for coeff in Poly(self.objective).coeffs()]), np.zeros(self.b.size))).astype(np.float)

        self.A = []
        A_slacks = np.identity(self.b.size)
        for i, constraint in enumerate(self.constraints):
            l = []
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            for name in self.variables:
                for id, symbol in self.variables[name].items(): # TODO name here instead of "x"?
                    l.append(coeff_dict.get(symbol, 0))
                self.A.append(l)
                
                if (constraint.comparator == GREATER_EQUAL):
                    A_slacks[:,i] *= -1
        self.A = np.hstack((np.array(self.A), A_slacks)).astype(np.float)
        print("self.A, self.b, self.c")
        print(self.A, self.b, self.c)

        if first:
            x_initial = self.get_initial_feasible_solution()
        else:
            x_initial = self.compute_feasible_basis()
         

        self.x, self.path, self.iterations = interior_point.interior_point(self.A, self.c, x_initial)

        return self.x

    def is_feasible(self, x):
        return all(np.isclose(self.A @ x, self.b, atol=10e-5))

    def get_initial_feasible_solution(self):
        amount_vars_slack = self.b.size
        amount_vars = self.c.size - self.b.size

        x = np.zeros(self.c.size)
        possible_items = []
        for i in range(1, 10, 1):
            possible_items += [i for _ in range(amount_vars)]
        permutations = itertools.permutations(possible_items, amount_vars)
        print("permutations")
        print(permutations)
        
        i = 0
        for x_test in permutations:
            x1 = np.hstack((np.array(x_test), np.zeros(amount_vars_slack)))
            print("\nx_test\n", x_test)
            print("\nself.b\n", self.b)
            print("\nself.A\n", self.A)
            print("\nx1\n", x1)
            x = np.hstack((np.array(x_test), self.b - (self.A @ x1)))

            if (self.is_feasible(x)):
                break
        if (not self.is_feasible(x)):
            raise Exception("No feasible solution found")

            i += 1
        print("Initial feasible solution: " + str(x))
        
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


#    def __repr__(self):
#        # Implement tableau
#        pass

