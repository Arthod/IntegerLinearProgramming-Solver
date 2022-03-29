import linear_programming as LP



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
    x3 = lp.add_variable(3)
    x4 = lp.add_variable(4)

    lp.set_objective(LP.MAX, 20 * x1 + 12 * x2 + 40 * x3 + 25 * x4)
    
    lp.add_constraint(x1 + x2 + x3 + x4, LP.LESS_EQUAL, 50)
    lp.add_constraint(3 * x1 + 2 * x2 + x3, LP.LESS_EQUAL, 100)
    lp.add_constraint(x2 + 2 * x3 + 3 * x4, LP.LESS_EQUAL, 90)

    #lp.add_constraint(x1, LP.GREATER_EQUAL, 0)
    #lp.add_constraint(x2, LP.GREATER_EQUAL, 0)
    #lp.add_constraint(x3, LP.GREATER_EQUAL, 0)
    #lp.add_constraint(x4, LP.GREATER_EQUAL, 0)

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

    lp.solve()