import unittest
import linear_programming as LP
import numpy as np

class TestLinearProgramming(unittest.TestCase):

    def test_solve(self):

        # Test, 2 variables 5 constraints
        lp = LP.LP()
        x1 = lp.add_variable(1)
        x2 = lp.add_variable(2)

        lp.set_objective(LP.MAX, 4*x1 - 2*x2)
        lp.add_constraint(x1 + 2 * x2,   LP.LESS_EQUAL, 8)
        lp.add_constraint(3 * x2,          LP.LESS_EQUAL, 8)
        lp.add_constraint(3 * x1 - 3 * x2,   LP.LESS_EQUAL, 8)
        lp.add_constraint(x1, LP.GREATER_EQUAL, 0)
        lp.add_constraint(x2, LP.GREATER_EQUAL, 0)

        x = lp.solve()
        z = x @ lp.c
        z_true = 14.22
        self.assertAlmostEqual(z, z_true, places=2)
        self.assertAlmostEqual(lp.A.tolist(), [
                [1, 2, 1, 0, 0, 0, 0],
                [0, 3, 0, 1, 0, 0, 0],
                [3, -3, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1]
            ])
        self.assertAlmostEqual(lp.b.tolist(), [8, 8, 8, 0, 0])
        self.assertAlmostEqual(lp.c.tolist(), [4, -2, 0, 0, 0, 0, 0])

        # Test, 4 variables 1 constraint
        lp = LP.LP()
        x1 = lp.add_variable(1)
        x2 = lp.add_variable(2)
        x3 = lp.add_variable(3)
        x4 = lp.add_variable(4)
        lp.set_objective(LP.MAX, 20 * x1 + 12 * x2 + 40 * x3 + 25 * x4)
        lp.add_constraint(x1 + x2 + x3 + x4, LP.LESS_EQUAL, 50)
        
        x = lp.solve()
        z = x @ lp.c
        z_true = 2000
        self.assertAlmostEqual(z, z_true, places=2)
        self.assertAlmostEqual(lp.A.tolist(), [
                [1, 1, 1, 1, 1]
            ])
        self.assertAlmostEqual(lp.b.tolist(), [50])
        self.assertAlmostEqual(lp.c.tolist(), [20, 12, 40, 25, 0])

        # Test, 2 variables 4 constraints
        lp = LP.LP()
        x1 = lp.add_variable(1)
        x2 = lp.add_variable(2)
        lp.set_objective(LP.MAX, x1 + 2 * x2)
        lp.add_constraint(1 * x1 + 1 * x2, LP.LESS_EQUAL, 21)
        lp.add_constraint(2 * x2, LP.LESS_EQUAL, 14)
        lp.add_constraint(x1 + 2 * x2, LP.LESS_EQUAL, 25)
        lp.add_constraint(2 * x1 + 9 * x2, LP.LESS_EQUAL, 80)
        
        x = lp.solve()
        z = x @ lp.c
        z_true = 25

        self.assertAlmostEqual(z, z_true, places=2)
        self.assertAlmostEqual(lp.A.tolist(), [
                [1, 1, 1, 0, 0, 0],
                [0, 2, 0, 1, 0, 0],
                [1, 2, 0, 0, 1, 0],
                [2, 9, 0, 0, 0, 1]
            ])
        self.assertAlmostEqual(lp.b.tolist(), [21, 14, 25, 80])
        self.assertAlmostEqual(lp.c.tolist(), [1, 2, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()