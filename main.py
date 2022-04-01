import linear_programming as LP
import itertools


if __name__ == "__main__":
    
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


    lp = LP.LP()
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    lp.set_objective(LP.MAX, x1 + 2 * x2)
    lp.add_constraint(x1 + x2, LP.LESS_EQUAL, 8)

    """
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    x3 = lp.add_variable(3)
    x4 = lp.add_variable(4)

    lp.set_objective(LP.MIN, 20 * x1 + 12 * x2 + 40 * x3 + 25 * x4)
    
    lp.add_constraint(x1 + x2 + x3 + x4, LP.LESS_EQUAL, 502)
    lp.add_constraint(- x1 - x2 - x3 - x4, LP.LESS_EQUAL, -5)
    lp.add_constraint(3 * x1 + 2 * x2 + x3, LP.LESS_EQUAL, 3000)
    lp.add_constraint(x2 + 2 * x3 + 3 * x4, LP.LESS_EQUAL, 90)
    """

    """
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)

    lp.set_objective(LP.MAX, 4*x1 - 2*x2)
    lp.add_constraint(x1 + 2 * x2,   LP.LESS_EQUAL, 2)
    lp.add_constraint(3 * x2,          LP.LESS_EQUAL, 2)
    lp.add_constraint(3 * x1 - 3 * x2,   LP.LESS_EQUAL, 2)
    lp.add_constraint(x1, LP.GREATER_EQUAL, 0)
    lp.add_constraint(x2, LP.GREATER_EQUAL, 0)
    """

    """
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    x3 = lp.add_variable(3)

    lp.set_objective(LP.MAX, 10*x1 + 15*x2 + 25*x3)
    lp.add_constraint(x1 + x2       + x3,   LP.LESS_EQUAL, 1000)
    lp.add_constraint(x1 - 2 * x2,          LP.LESS_EQUAL, 0)
    lp.add_constraint(                x3,   LP.LESS_EQUAL, 340)
    lp.add_constraint(x1, LP.GREATER_EQUAL, 0)
    lp.add_constraint(x2, LP.GREATER_EQUAL, 0)
    lp.add_constraint(x3, LP.GREATER_EQUAL, 0)
    """

    x = lp.solve()
    lp.plot_solution_path()

    """
    def add(a, b):
        return a + b
    def mult(a, b):
        return a * b
    def sub(a, b):
        return a - b
    def div(a, b):
        return a / b

    print(list(itertools.permutations([1,5,6,7])))
    for nrs in list(itertools.permutations([1,5,6,7])):
        for operation1 in [add, mult, sub, div]:
            for operation2 in [add, mult, sub, div]:
                for operation3 in [add, mult, sub, div]:
                    try:
                        n = operation1(nrs[0], operation2(nrs[1], operation3(nrs[2], nrs[3])))
                        if (n == 21):
                            print("now")
                            print(operation1, operation2, operation3)
                            print(nrs)
                    except:
                        pass
    """