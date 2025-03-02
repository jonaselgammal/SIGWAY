# Global
import os
import unittest
import numpy as np
import jax.numpy as jnp
from scipy.integrate import simpson

# Local
from sigway import utils as ut


# Load test data
test_data = np.load(
    os.path.join(os.path.dirname(__file__), "test_data/data_test_utils.npz")
)
# These are values of N, H, and k_vals for chaotic inflation N
# are 61 linealy spaced values from 0 to 60 the dynamics of the model assumes
# the lowest order slow-roll approximation, so the Hubble parameter is
# calculated as H = V / (2 * pi * dV / dphi) where V is the potential and
# dV / dphi is the derivative of the potential with respect to the inflaton
# field phi. The potential is V = 0.5 * m**2 * phi**2 and the derivative is
# dV / dphi = m**2 * phi, where m is the mass of the inflaton field.
N, H, k_vals = test_data["data_1"]

# These are some values of k and H for chaotic inflation
kk, HH = test_data["data_2"]


class TestUnits(unittest.TestCase):

    def test_wavenumber_from_efolds_si_units(self):
        """
        Test the function wavenumber_from_efolds_si_units from utils.py.

        """

        # compute the wavenumber ks from the e-folds N and H
        ks = ut.wavenumber_from_efolds_si_units(N, H, 60, H[-1])

        # check that the output is correct
        self.assertEqual(np.sum(1 - k_vals / ks), 0.0)

    def test_efolds_from_wavenumber_si_units(self):
        """
        Test the function efolds_from_wavenumber_si_units from utils.py.

        """

        # compute the wavenumber ks from the e-folds N and H
        NN = ut.efolds_from_wavenumber_si_units(k_vals, H, 60, H[-1])

        # check that the output is correct
        self.assertEqual(np.sum(NN - N), 0.0)

    def test_H_from_wavenumber(self):
        """
        Test the function H_from_wavenumber from utils.py.
        """

        # compute H for some values of the the wavenumber kk
        H_k = ut.H_from_wavenumber(kk, N, H, 60, H[-1])

        # check that the output is correct
        self.assertEqual(np.sum(1 - H_k / HH), 0.0)

    def test_simpson_uniform(self):
        """
        Test the function simpson_uniform from utils.py.
        """

        # check with even number of points
        x1 = jnp.linspace(0.0, 1.0, 100)
        y1 = x1**2

        res_1 = simpson(y1, x=x1)
        res_2 = ut.simpson_uniform(y1, x1)

        print(len(x1), res_1, res_2)

        # check that the output is correct
        self.assertAlmostEqual(res_1 / res_2, 1.0, places=15)

        # check with odd number of points
        x1 = jnp.linspace(0.0, 1.0, 101)
        y1 = x1**3

        res_1 = simpson(y1, x=x1)
        res_2 = ut.simpson_uniform(y1, x1)

        print(len(x1), res_1, res_2)

        # check that the output is correct
        self.assertAlmostEqual(res_1 / res_2, 1.0, places=15)

    def test_simpson_nonuniform(self):
        """
        Test the function simpson_nonuniform from utils.py.
        """

        # check with even number of points
        x1 = jnp.geomspace(1.0, 10.0, 100)
        y1 = x1**2

        res_1 = simpson(y1, x=x1)
        res_2 = ut.simpson_nonuniform(y1, x1)

        print(len(x1), res_1, res_2)

        # check that the output is correct
        self.assertAlmostEqual(res_1 / res_2, 1.0, places=15)

        # check with odd number of points
        x1 = jnp.geomspace(1.0, 10.0, 101)
        y1 = x1**3

        res_1 = simpson(y1, x=x1)
        res_2 = ut.simpson_nonuniform(y1, x1)

        print(len(x1), res_1, res_2)

        # check that the output is correct
        self.assertAlmostEqual(res_1 / res_2, 1.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
