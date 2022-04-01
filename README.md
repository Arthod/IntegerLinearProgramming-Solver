# IntegerLinearProgramming-Solver (WIP)

Solver for (Integer) Linear Programming Problems

```
if __name__ == "__main__":
    lp = LP.LP()
    x1 = lp.add_variable(1)
    x2 = lp.add_variable(2)
    lp.set_objective(LP.MAX, x1 + 2 * x2)
    lp.add_constraint(1 * x1 + 1 * x2, LP.LESS_EQUAL, 21)
    lp.add_constraint(2 * x2, LP.LESS_EQUAL, 14)
    lp.add_constraint(x1 + 2 * x2, LP.LESS_EQUAL, 25)
    lp.add_constraint(2 * x1 + 9 * x2, LP.LESS_EQUAL, 80)


    x = lp.solve()
    print("z = " + str(x @ lp.c))
    print("x = " + str(x))
    lp.plot_solution_path()
```
