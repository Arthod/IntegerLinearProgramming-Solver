import copy
import itertools
import scipy
import sympy
import numpy as np
from scipy.optimize import linprog
from typing import Dict # for type hinting

import interior_point
import simplex



GREATER_EQUAL = -1
EQUAL = 0
LESS_EQUAL = 1
COMPARATORS = {
    EQUAL: "=",
    GREATER_EQUAL: ">=",
    LESS_EQUAL: "<="
    }

MAX = 1
MIN = -1

SOLVER_SIMPLEX = "simplex"
SOLVER_INTERIOR_POINT_METHOD = "interiorPointMethod"

class Constraint:
    def __init__(self, expr_LHS, comparator, expr_RHS):
        self.expr_LHS = expr_LHS
        self.comparator = comparator
        self.comparator_prev = None
        self.expr_RHS = expr_RHS

    def __repr__(self):
        return str(self.expr_LHS) + COMPARATORS[self.comparator] + str(self.expr_RHS)


class LinearProgrammingProblem:
    def __init__(self):
        # Problem definition
        self.variables: Dict[str, sympy.Symbol] = {}
        self.constraints: list[Constraint] = []
        self.is_maximizing: bool = None
        self.objective: sympy.core.Expr = None

        # Flags
        self._A_updated = False
        self._b_updated = False
        self._c_updated = False

        # Matricies
        self._A = None
        self._b = None
        self._c = None

        self.vars_slack_amount = 0
        self.vars_artificial_amount = 0

    @staticmethod
    def parse(problem_str: str) -> "LinearProgrammingProblem":
        def parse_equation(equation_str: str, variables: Dict[str, sympy.core.Symbol]) -> sympy.core.Expr:
            """
            900x1 + 1400x2 + 700x3 + 1000x4 + 1700x5 + 900x6
            900x1 + 1400x2 + 700x3 + 1000 * x4 + 1700x5 + 900x6
            x4 + x5 + x6
            x + y + z
            """
            equation = 0
            # Add "*" between symbols and coefficient
            equation_str = equation_str.replace(" ", "").replace("*", "")
            equation_str_parts = equation_str.replace("-", "+").split("+")
            operations = [s for s in equation_str if s == "+" or s == "-"]
            if (equation_str[0] != "-" and equation_str[0] != "+"):
                operations.insert(0, "+")
            for i, equation_str_part in enumerate(equation_str_parts):
                for variable_key, variable in variables.items():
                    if (variable_key in equation_str_part):
                        coefficient = equation_str_part.replace(variable_key, "")
                        if (coefficient == ""):
                            coefficient = 1
                        else:
                            coefficient = float(coefficient)
                        if (operations[i] == "+"):
                            equation = equation + coefficient * variable
                        elif (operations[i] == "-"):
                            equation = equation - coefficient * variable

            return equation

        lp = LinearProgrammingProblem()

        problem_list = []
        for p in problem_str.split("\n"):
            if (p.strip() != ""):
                problem_list.append(p.strip())
        
        assert(len(problem_list) >= 2), "Format error: Atleast one constraint needed."
        
        ## Objective function and variables
        # Max or min?
        objective_function_str = problem_list[0].split("=")
        is_maximizing = None
        if ("min" in objective_function_str[0].lower()):
            is_maximizing = False
        elif ("max" in objective_function_str[0].lower()):
            is_maximizing = True
        assert is_maximizing is not None, "Format error: Ambigious whether to maximize or minimize."

        # Variables
        vars_possible = [symbol + str(i) for i in range(10) for symbol in ["a", "b", "c", "d", "x", "y", "z", "w"]] + ["a", "b", "c", "d", "x", "y", "z", "w"]
        variables = {}
        for var in vars_possible:
            if (var in objective_function_str[1]):
                variables[var] = lp.add_variable(var)
        assert (len(variables)), "Format error: Atleast one variable needed."

        # Objective function
        objective_function_expr = parse_equation(objective_function_str[1].replace("subject to", ""), variables)
        lp.set_objective(is_maximizing, objective_function_expr)

        # Constraints
        for p in problem_list[1:]:
            comparator = None
            RHS = None
            LHS = None

            # Comparator
            for comparator_name, comparator_symbol in COMPARATORS.items():
                if (comparator_symbol in p):
                    comparator = comparator_name
            assert comparator is not None, "Format error: Invalid comparator."

            # LHS & RHS
            p = p.split(COMPARATORS[comparator])
            assert len(p) == 2, "Format error: Invalid constraint"
            LHS = parse_equation(p[0], variables)
            RHS = float(p[1])
            
            lp.add_constraint(LHS, comparator, RHS)

        return lp


        




    def __repr__(self):
        return f"A:\n{str(self.A)},\nb:\n{str(self.b)},\nc:\n{str(self.c)})"

    @property
    def A(self) -> np.array:
        if (not self._A_updated):
            A = []
            for _, constraint in enumerate(self.constraints):
                #arr = []
                coeff_dict = constraint.expr_LHS.as_coefficients_dict()
                #for _, symbol in self.variables.items(): # TODO name here instead of "x"?
                #    arr.append(coeff_dict.get(symbol, 0))
                #A.append(arr)
                A.append([coeff_dict.get(symbol, 0) for _, symbol in self.variables.items()])

            self._A = np.array(A).astype(np.float)
            self._A_updated = True

        return self._A

    @property
    def c(self) -> np.array:
        if (not self._c_updated):
            coeff_dict = self.objective.as_coefficients_dict()

            if (self.is_maximizing): mlt = 1
            else: mlt = -1

            c = [coeff_dict.get(symbol, 0) * mlt for _, symbol in self.variables.items()]

            self._c = np.array(c).astype(np.float)
            self._c_updated = True
        return self._c

    @property
    def b(self) -> np.array:
        if (not self._b_updated):
            self._b = np.array([constraint.expr_RHS for constraint in self.constraints])
            self._b_updated = True

        return self._b

    def copy(self) -> "LinearProgrammingProblem":
        lp = LinearProgrammingProblem()

        lp.variables = copy.deepcopy(self.variables)
        lp.constraints = copy.deepcopy(self.constraints)
        lp.is_maximizing = self.is_maximizing
        lp.objective = copy.deepcopy(self.objective)

        lp._A_updated = False
        lp._b_updated = False
        lp._c_updated = False

        return lp
        

    def slacken_problem(self) -> "LinearProgrammingProblem":
        lp_slacked = self.copy()

        for i, constraint in enumerate(lp_slacked.constraints):
            if (constraint.comparator != EQUAL):
                slack_variable = lp_slacked.add_variable(i, name="_0slack")
                constraint.expr_LHS += constraint.comparator * slack_variable

                lp_slacked.vars_slack_amount += 1
                constraint.comparator_prev = constraint.comparator
                constraint.comparator = EQUAL

        return lp_slacked


    def add_variable(self, key: any, name: str="x") -> sympy.Symbol:
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
        key = f"{name}_{key}"
        assert key not in self.variables, f"Variable named '{key}' already used"

        # Otherwise add the key to the variable group
        variable = sympy.Symbol(key)
        self.variables[key] = variable

        return variable


    def add_constraint(self, expr_LHS: sympy.core.Expr, comparator: int, expr_RHS: sympy.core.Expr) -> None:
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

        self.constraints.append(Constraint(expr_LHS, comparator, expr_RHS))

        # Update flags
        self._A_updated = False
        self._b_updated = False

    def set_objective(self, is_maximizing: bool, expr: sympy.core.Expr):
        self.is_maximizing = is_maximizing
        self.objective = expr

        # Update flags
        self._c_updated = False

    def compute_feasible_basis(self):
        # LP for finding feasible basis
        F_LP = LinearProgrammingProblem()
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
        F_LP.set_objective(MAX, -sum(artificial_variables) * self.is_maximizing)

        # Solve feasibility LP and remove introduces variables
        x_feasible = F_LP._solve()
        
        # Return if feasible
        assert F_LP.is_feasible(x_feasible), "Didn't find feasible basis"

        return x_feasible

    @staticmethod
    def is_feasible(lp: "LinearProgrammingProblem", x: np.array, tol=10e-5) -> bool:
        return all(np.isclose(lp.A @ x, lp.b, atol=tol))

    def solve(self, method=None):
        lp_slacked = self.slacken_problem()
        # Decide which solver to use
        # Simplex?
        if (method == None):
            x_zero = simplex.zero_point_solution(lp_slacked.A, lp_slacked.b, lp_slacked.c, len(self.variables))
            if (simplex.is_feasible(lp_slacked.A, x_zero, lp_slacked.b)):
                # If zero point solution is feasible, use simplex
                method = SOLVER_SIMPLEX

        if (method == None):
            # Solve using interior point algorithm
            x_ones = np.ones(len(self.variables))
            x_initial = np.hstack((x_ones, self.b - (self.A @ x_ones)))
            if (self.is_feasible(self, x_initial)):
                # IPA is feasible
                method = SOLVER_INTERIOR_POINT_METHOD


        x_sol = None
        iterations_count = None

        if (method == SOLVER_SIMPLEX):
            # Solve using simplex
            x_sol, iterations_count = simplex.simplex(lp_slacked.A, lp_slacked.b, lp_slacked.c, lp_slacked.vars_slack_amount)
            
            # Cut slack variables
            x_sol = x_sol[:lp_slacked.c.size - lp_slacked.vars_slack_amount]

        elif (method == SOLVER_INTERIOR_POINT_METHOD):
            # Solve using interior point algorithm
            x_ones = np.ones(len(self.variables))
            x_initial = np.hstack((x_ones, self.b - (self.A @ x_ones)))

            #x_initial = np.array([1 for _ in range(len(lp_slacked.variables))])
            x_sol, path, iterations_count = interior_point.interior_point(lp_slacked.A, lp_slacked.c, x_initial)

            # Cut slack variables
            x_sol = x_sol[:lp_slacked.c.size - lp_slacked.vars_slack_amount]

        else:
            print("No solution found")
            return

        print(f"Solving using {method}")
        print(f"Solved in {iterations_count} iterations")
        print("z = " + str(x_sol @ self.c))
        print("x = " + str(x_sol))
        
        return x_sol


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

