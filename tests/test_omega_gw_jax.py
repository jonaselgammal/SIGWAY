# Global
import os
import unittest
import numpy as np

# Local
from sigway import omega_gw_jax as og
from sigway.kernels import I_sq_RD_uv

# Load test data
test_data = np.load(
    os.path.join(
        os.path.dirname(__file__), "test_data/data_test_omega_gw_jax.npz"
    )
)
s = test_data["s"]
t = test_data["t"]
u, v, pol = test_data["data_1"]
I_sq_RD = test_data["I_sq_RD"]


class TestUnits(unittest.TestCase):

    def test_get_u(self):
        """
        Test the function get_u from omega_gw_jax.py.

        """

        # compute u from some values of s and t
        uu = og.get_u(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(u - uu), 0.0)

    def test_get_v(self):
        """
        Test the function get_v from omega_gw_jax.py.

        """

        # compute v from some values of s and t
        vv = og.get_v(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(v - vv), 0.0)

    def test_polynomial(self):
        """
        Test the function polynomial from omega_gw_jax.py.
        """

        # compute pol from some values of s and t
        polynomial = og.polynomial(t[:, None], s[None, :])

        # check that the output is correct
        self.assertEqual(np.sum(pol - polynomial), 0.0)

    def test_I_sq_RD(self):
        """
        Test the function I_sq_RD from omega_gw_jax.py.
        """
        # compute pol from some values of s and t
        I_sq_RD_v = og.I_sq_RD(t[:, None], s[None, :], k=1.0)

        # Compare to the stored snapshot with a relative tolerance: the kernel's
        # log/heaviside make it sensitive to per-version floating-point rounding
        # (exact equality fails by ~1e-12 on numpy/XLA builds other than the one
        # the snapshot was generated on). The physics is pinned independently in
        # test_kernels_rd.py against the textbook closed form.
        scale = float(np.max(np.abs(I_sq_RD)))
        self.assertTrue(
            np.allclose(I_sq_RD_v, I_sq_RD, rtol=1e-7, atol=1e-9 * scale)
        )

    def test_I_sq_RD_vs_uv(self):
        """
        Test that the two function I_sq_RD and I_sq_RD_uv give the same output
        (up to numerical precision) in a reasonable range for t.
        """
        tt = np.geomspace(0.01, 100, 10)

        # compute I_sq_RD from some values of s and t
        I_sq_RD = og.I_sq_RD(tt[:, None], s[None, :], k=1.0)

        I_sq_RD_2 = I_sq_RD_uv(tt[:, None], s[None, :], k=1.0)

        # check that the output is correct
        self.assertAlmostEqual(np.sum(I_sq_RD / I_sq_RD_2 - 1), 0.0, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
