import unittest
import linear_programming as LP
import numpy as np


class TestLinearProgramming(unittest.TestCase):
    def assertArrayAlmostEqual(self, arr1, arr2):
        self.assertEqual(len(arr1), len(arr2))
        for i in range(len(arr1)):
            self.assertAlmostEqual(arr1[i], arr2[i])

    def test_solve(self):

        """ Test
        Maximize p = x + 2y + 3z subject to 
            x + y <= 20.5
            y + z <= 20.5
            x + z <= 30.5
        """
        lp = LP.LP()
        x = lp.add_variable(1)
        y = lp.add_variable(2)
        z = lp.add_variable(3)
        lp.set_objective(LP.MAX, x + 2 * y + 3 * z)
        lp.add_constraint(x + y, LP.LESS_EQUAL, 20.5)
        lp.add_constraint(y + z, LP.LESS_EQUAL, 20.5)
        lp.add_constraint(x + z, LP.LESS_EQUAL, 30.5)

        x = lp.solve()
        z = x @ lp.c
        z_true = 71.5
        self.assertAlmostEqual(z, z_true, places=2)
        self.assertArrayAlmostEqual(lp.A.tolist(), [
                [1, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 1]
            ])
        print(lp.b.tolist())
        self.assertArrayAlmostEqual(lp.b.tolist(), [20.5, 20.5, 30.5])
        self.assertArrayAlmostEqual(lp.c.tolist(), [1, 2, 3, 0, 0, 0])


        """
        Maximize p = (1/2)x + 3y + z + 4w subject to 
            x + y + z + w <= 40
            2x + y - z - w >= 10
            w - y >= 12
        """
        lp = LP.LP()
        x = lp.add_variable(1)
        y = lp.add_variable(2)
        z = lp.add_variable(3)
        w = lp.add_variable(4)
        lp.set_objective(LP.MAX, (1/2) * x + 3 * y + z + 4 * w)
        lp.add_constraint(x + y + z + w, LP.LESS_EQUAL, 40)
        lp.add_constraint(2 * x + y - z - w, LP.GREATER_EQUAL, 10)
        lp.add_constraint(w - y, LP.GREATER_EQUAL, 12)

        x = lp.solve()
        z = x @ lp.c
        z_true = 113
        self.assertAlmostEqual(z, z_true, places=2)

if __name__ == '__main__':
    unittest.main()