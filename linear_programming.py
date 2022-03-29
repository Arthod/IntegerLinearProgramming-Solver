from sympy import Symbol


EQUAL = 0
GREATER_EQUAL = 1
LESS_EQUAL = 2
COMPARATORS = {
    EQUAL: "==",
    GREATER_EQUAL: ">=",
    LESS_EQUAL: "<="
    }


class LP:
    def __init__(self):
        self.variables = {}
        self.constraints = []

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


    def add_constraint(self, constr):
        self.constraints.append(constr)

    """
    def add_constraint(self, expr_LHS: str, comparator: int, expr_RHS):
        assert comparator in COMPARATORS

        if (comparator == EQUAL):
            constr = expr_LHS == expr_RHS
        if (comparator == GREATER_EQUAL):
            constr = expr_LHS >= expr_RHS
        if (comparator == LESS_EQUAL):
            constr = expr_LHS <= expr_RHS

        self.constraints.append(constr)

    """