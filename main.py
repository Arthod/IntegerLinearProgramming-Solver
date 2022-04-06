import linear_programming as LP


if __name__ == "__main__":    

    lp = LP.LinearProgrammingProblem()
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    lp.set_objective(LP.MAX, 6 * x1 + 8 * x2)
    lp.add_constraint(5 * x1 + 10 * x2, LP.LESS_EQUAL, 60)
    lp.add_constraint(4 * x1 + 4 * x2, LP.LESS_EQUAL, 40)
    
    """
    lp = LP.LP()
    x = lp.add_variable(1)
    y = lp.add_variable(2)
    z = lp.add_variable(3)
    lp.set_objective(LP.MAX, x + 2 * y + 3 * z)
    lp.add_constraint(x + y, LP.LESS_EQUAL, 20.5)
    lp.add_constraint(y + z, LP.GREATER_EQUAL, 20.5)
    lp.add_constraint(x + z, LP.GREATER_EQUAL, 30.5)
    """

    
    x = lp.solve()
    
    print("z = " + str(x @ lp.c))
    print("x = " + str(x))
    print("iterations = " + str(lp.iterations))
    if (len(lp.variables) == 2):
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