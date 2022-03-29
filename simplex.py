import linear_programming as LP


if __name__ == "__main__":
    lp = LP.LP()

    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    x3 = lp.add_variable(3)
    x4 = lp.add_variable(4)

    lp.add_constraint(x1 + x2 + x3 + x4 == 3)

    print(lp.constraints)