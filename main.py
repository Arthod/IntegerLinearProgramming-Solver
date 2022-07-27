import linear_programming as LP



if __name__ == "__main__":    

    lp = LP.LinearProgrammingProblem()
    """
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    lp.set_objective(LP.MAX, 6 * x1 + 8 * x2)
    lp.add_constraint(5 * x1 + 10 * x2, LP.LESS_EQUAL, 60)
    lp.add_constraint(4 * x1 + 4 * x2, LP.LESS_EQUAL, 40)
    """

    """
    x = lp.add_variable(1)
    y = lp.add_variable(2)
    z = lp.add_variable(3)
    lp.set_objective(LP.MAX, x + 2 * y + 3 * z)
    lp.add_constraint(x + y, LP.LESS_EQUAL, 20.5)
    lp.add_constraint(y + z, LP.LESS_EQUAL, 20.5)
    lp.add_constraint(x + z, LP.LESS_EQUAL, 32.5)
    lp.solve()
    """

    """
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    x3 = lp.add_variable(3)
    x4 = lp.add_variable(4)
    x5 = lp.add_variable(5)
    x6 = lp.add_variable(6)
    lp.set_objective(LP.MIN, 900 * x1 + 1400 * x2 + 700 * x3 + 1000 * x4 + 1700 * x5 + 900 * x6)
    lp.add_constraint(x1 + x2 + x3, LP.LESS_EQUAL, 10)
    lp.add_constraint(x4 + x5 + x6, LP.LESS_EQUAL, 10)
    lp.add_constraint(x1 + x4, LP.GREATER_EQUAL, 6)
    lp.add_constraint(x2 + x5, LP.GREATER_EQUAL, 4)
    lp.add_constraint(x3 + x6, LP.GREATER_EQUAL, 4)
    lp.add_constraint(x4 - x6, LP.LESS_EQUAL, 0)
    """



    lp = LP.LinearProgrammingProblem.parse("""
        Maximize p = 1x + 3y + z + 4w subject to 
        x + y + z + w <= 40
        2x + y - z - w >= 10
        w - y >= 12
    """)
    #w = lp.add_variable("w")
    #x = lp.variables["x_x"]
    #lp.add_constraint(w + x, LP.LESS_EQUAL, 10)
    
    lp.solve(method=LP.SOLVER_INTERIOR_POINT_METHOD)

    #lp = LP.LinearProgrammingProblem.parse("""
    #    Maximize p = 1x + 3y + z + 4w subject to 
    #    x + y + z + w <= 40
    #    2x + y - z - w >= 10
    #    w - y >= 12
    #""")
    #print(lp.variables)
    #print(lp.constraints)
    #lp.solve()




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