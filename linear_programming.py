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

        self.vars_slack_amount = 0
        self.vars_artificial_amount = 0


    def compute_feasible_basis(self):
        # LP for finding feasible basis
        F_LP = LP()
        F_LP.vars_slack_amount = self.vars_slack_amount
        
        # Get variables from original problem
        F_LP.variables = self.variables.copy()

        # Get contraints from original problem
        F_LP.constraints = self.constraints.copy()
        
        # Add artificial variables, and add them to constraints
        artificial_variables = []
        for i, constraint in enumerate(F_LP.constraints):
            #if (constraint.comparator_prev != LESS_EQUAL):
            artificial_variable = F_LP.add_variable(i, name="_1artificial")
            artificial_variables.append(artificial_variable)

            constraint.expr_LHS += artificial_variable
            F_LP.vars_artificial_amount += 1
            #if constraint.expr_RHS < 0:
            #    constraint.expr_LHS *= -1
            #    constraint.expr_RHS *= -1
        
        
        # Objective function of feasibility LP
        F_LP.set_objective(self.is_maximizing, -sum(artificial_variables))

        # Solve feasibility LP and remove introduces variables
        x_feasible = F_LP._solve()
        
        # Return if feasible
        assert F_LP.is_feasible(x_feasible), "Didn't find feasible basis"

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

        # If key already in the group of the variable we raise an exception
        key = name + "_" + str(key)
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

    def compute_c(self):
        coeff_dict = self.objective.as_coefficients_dict()
        c = []
        for _, symbol in self.variables.items(): # TODO name here instead of "x"?
            c.append(coeff_dict.get(symbol, 0) * self.is_maximizing)
        self.c = np.array(c).astype(np.float)

    def set_objective(self, is_maximizing, expr):
        self.is_maximizing = is_maximizing
        self.objective = expr

    def solve(self):
        x = self._solve(lp_is_raw=True)

        # Throw away slack solution
        x = x[:len(self.variables)]

        self.compute_c()

        return x

    def _solve(self, lp_is_raw: bool=False):
        lp = self
        if (lp_is_raw): # if there are no slack or artificial vars
            # Copy the problem LP and add slack variables
            lp = lp_slack = LP()

            # Objective
            lp_slack.set_objective(self.is_maximizing, self.objective)

            # Get variables from original problem
            lp_slack.variables = self.variables.copy()

            # Get contraints from original problem
            lp_slack.constraints = self.constraints.copy()

            # Add slack variables where constraint isn't EQUAL.
            # And set constraint to EQUAL
            slack_variables = []
            for i, constraint in enumerate(lp_slack.constraints):
                if (constraint.comparator != EQUAL):
                    slack_variable = lp_slack.add_variable(i, name="_0slack")
                    slack_variables.append(slack_variable)
                    constraint.expr_LHS += constraint.comparator * slack_variable

                    lp_slack.vars_slack_amount += 1
                    constraint.comparator_prev = constraint.comparator
                    constraint.comparator = EQUAL

        lp.b = np.array([constraint.expr_RHS for constraint in lp.constraints])
        
        coeff_dict = lp.objective.as_coefficients_dict()
        c = []
        for _, symbol in lp.variables.items(): # TODO name here instead of "x"?
            c.append(coeff_dict.get(symbol, 0) * lp.is_maximizing)
        lp.c = np.array(c).astype(np.float)

        lp.A = []
        for i, constraint in enumerate(lp.constraints):
            arr = []
            coeff_dict = constraint.expr_LHS.as_coefficients_dict()
            for _, symbol in lp.variables.items(): # TODO name here instead of "x"?
                arr.append(coeff_dict.get(symbol, 0))
            lp.A.append(arr)
        lp.A = np.array(lp.A).astype(np.float)

        if (lp_is_raw):
            x_initial = lp.compute_feasible_basis()

            # Throw away artificial solution
            x_initial = x_initial[:len(lp.variables)]

        else:
            x_initial = self.get_initial_feasible_solution()
         

        lp.x, lp.path, lp.iterations = interior_point.interior_point(lp.A, lp.c, x_initial)

        return lp.x

    def is_feasible(self, x):
        return all(np.isclose(self.A @ x, self.b, atol=10e-5))

    def get_initial_feasible_solution(self):
        amount_vars_slack = self.vars_slack_amount
        amount_vars = self.c.size - self.vars_slack_amount - self.vars_artificial_amount

        x = np.zeros(self.c.size)
        possible_items = []
        for i in range(1, 10, 1):
            possible_items += [i for _ in range(amount_vars)]#amount_vars)]
        permutations = itertools.permutations(possible_items, amount_vars + amount_vars_slack)
        
        for x_test in permutations:
            x1 = np.hstack((np.array(x_test), np.zeros(self.vars_artificial_amount)))
            x = np.hstack((np.array(x_test), self.b - (self.A @ x1)))

            print(x)
            if (self.is_feasible(x)):
                if (np.min(self.b - (self.A @ x1)) <= 0):
                    continue
                break
        if (not self.is_feasible(x)):
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


class Constraint:
    def __init__(self, expr_LHS, comparator, expr_RHS):
        self.expr_LHS = expr_LHS
        self.comparator = comparator
        self.comparator_prev = None
        self.expr_RHS = expr_RHS

    def __repr__(self):
        return str(self.expr_LHS) + COMPARATORS[self.comparator] + str(self.expr_RHS)


#    def __repr__(self):
#        # Implement tableau
#        pass

