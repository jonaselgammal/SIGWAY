# Global
import os
import unittest
import numpy as np
import test

# Local
from sigway import omega_gw_jax as og


# Load test data
test_data = np.load(
    os.path.join(
        os.path.dirname(__file__), "test_data/data_test_omega_gw_jax.npz"
    )
)
s = test_data["s"]
t = test_data["t"]
u, v, pol = test_data["data_1"]


class TestUnits(unittest.TestCase):

    def test_get_u(self):
        """
        Test the function get_u from units.py.

        """

        # compute u from some values of s and t
        uu = og.get_u(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(u - u), 0.0)

    def test_get_v(self):
        """
        Test the function get_v from units.py.

        """

        # compute v from some values of s and t
        vv = og.get_v(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(v - v), 0.0)

    def test_polynomial(self):
        """
        Test the function polynomial from units.py.
        """

        # compute pol from some values of s and t
        polynomial = og.polynomial(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(pol - polynomial), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
