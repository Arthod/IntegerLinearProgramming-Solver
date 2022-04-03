import itertools
import random
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



class LP:
    def __init__(self):
        # Variables and constraints
        self.variables: dict[any, Symbol] = {}
        self.constraints: list[Constraint] = []

        # Problem definition
        self.is_maximizing = None
        self.objective = None
        self.iterations = 0

    @property
    def A(self) -> np.array:
        # Returns A (without slack variables)
        A = []
        for _, constraint in enumerate(self.constraints):
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            arr = []
            for _, symbol in self.variables.items(): # TODO name here instead of "x"?
                arr.append(coeff_dict.get(symbol, 0))
            A.append(arr)

        return np.array(A)

    @property
    def A_slacks(self) -> np.array:
        A_slacks = np.identity(len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            if (constraint.comparator == GREATER_EQUAL):
                A_slacks[:,i] *= -1

        return np.hstack((np.array(self.A), A_slacks)).astype(np.float)

    @property
    def b(self):
        return np.array([
            constraint.expr_RHS for constraint in self.constraints])

    @property
    def c(self):
        coeff_dict = self.objective.as_coefficients_dict()

        c = []
        for _, symbol in self.variables.items(): # TODO name here instead of "x"?
            c.append(coeff_dict.get(symbol, 0) * self.is_maximizing)
            
        return np.array(c).astype(np.float)
        #return np.array([self.is_maximizing * coeff for coeff in Poly(self.objective).coeffs()]).astype(np.float)

    @property
    def c_slacks(self):
        return np.hstack((
            self.c,
            np.zeros(len(self.constraints))
            )).astype(np.float)



    def compute_feasible_basis(self):
        # LP for finding feasible basis
        F_LP = LP()

        A_slacks = self.A_slacks
        
        # Get variables from original problem
        F_LP.variables = self.variables.copy()

        # Get contraints from original problem
        F_LP.constraints = self.constraints.copy()
        
        # Add FEAS variables, and add them to constraints
        feas_index = 1
        feas_variables = []
        for constraint in F_LP.constraints:
            if constraint.expr_RHS < 0:
                constraint.expr_LHS *= -1
                constraint.expr_RHS *= -1
            feas_variable = F_LP.add_variable(feas_index, name="FEAS")
            feas_variables.append(feas_variable)
            constraint.expr_LHS += feas_variable
            feas_index += 1
        
        # Objective function of feasibility LP
        F_LP.set_objective(MAX, -sum(feas_variables) * self.is_maximizing)

        # Solve feasibility LP and remove introduces variables
        x_feasible = F_LP.solve(first=True)
        print(x_feasible)

        # Throw away artificial solution 
        amount_vars_slack = len(F_LP.constraints)
        amount_vars = len(F_LP.variables) - len(F_LP.constraints)
        x_feasible = np.hstack((
            x_feasible[0:amount_vars],
            x_feasible[x_feasible.size - amount_vars_slack:x_feasible.size]
            ))
        print(x_feasible)

        # Return if feasible
        #assert self.is_feasible(self.A_slacks, self.b, x_feasible), "Didn't find feasible basis"
        return x_feasible


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
        key = f"{name}_{key}"

        # If key already in the group of the variable we raise an exception
        if (key in self.variables):
            raise Exception("variable key already used")

        # Otherwise add the key to the variable group
        symbol = Symbol(key)
        self.variables[key] = symbol

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

    def solve(self, first: bool=False):
        A_slacks = self.A_slacks
        b = self.b
        c_slacks = self.c_slacks

        amount_vars = c_slacks.size - b.size
        if (first):
            # Re-arrange the slack variables for the "get initial feasible solution calc,"
            amount_vars_slack = b.size

            A_slacks_rearranged = np.hstack((
                A_slacks[:,0:amount_vars - amount_vars_slack],
                A_slacks[:,amount_vars:A_slacks.size],
                A_slacks[:,amount_vars - amount_vars_slack:amount_vars]
                ))
            c_slacks_rearranged = np.hstack((
                c_slacks[0:amount_vars - amount_vars_slack],
                c_slacks[amount_vars:A_slacks.size],
                c_slacks[amount_vars - amount_vars_slack:amount_vars]
                ))

            x_initial = self.get_initial_feasible_solution(A_slacks_rearranged, b, c_slacks_rearranged)
        else:
            x_initial = self.compute_feasible_basis()
         
        print("initial feasible found: " + str(x_initial))

        self.x, self.path, self.iterations = interior_point.interior_point(A_slacks, c_slacks, x_initial)
        self.x = self.x[:amount_vars]

        return self.x

    def is_feasible(self, A, b, x):
        print(A, b)
        print(A @ x)
        return all(np.isclose(A @ x, b, atol=10e-3))

    def get_initial_feasible_solution(self, A_slacks, b, c_slacks):
        amount_vars_slack = b.size
        amount_vars = c_slacks.size - b.size

        x = np.zeros(c_slacks.size)
        possible_items = []
        for i in range(1, 10, 1):
            possible_items += [1 for _ in range(1)]#amount_vars)]
        permutations = itertools.permutations(possible_items, amount_vars)

        for x_test in permutations:
            x1 = np.hstack((np.array(x_test), np.zeros(amount_vars_slack)))
            x = np.hstack((np.array(x_test), b - (A_slacks @ x1)))
            
            if (self.is_feasible(A_slacks, b, x)):
                if (np.min(b - (A_slacks @ x1)) <= 0):
                    continue
                break
        if (not self.is_feasible(A_slacks, b, x)):
            raise Exception("No feasible solution found")
        
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
