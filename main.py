import linear_programming as LP


if __name__ == "__main__":    


    lp = LP.LP()
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    x3 = lp.add_variable(3)
    x4 = lp.add_variable(4)
    lp.set_objective(LP.MAX, 0.5 * x1 + 3 * x2 + x3 + 4 * x4)
    lp.add_constraint(x1 + x2 + x3 + x4, LP.LESS_EQUAL, 40)
    lp.add_constraint(2 * x1 + x2 - x3 - x4, LP.GREATER_EQUAL, 10)
    lp.add_constraint(x4 - x2, LP.GREATER_EQUAL, 12)

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