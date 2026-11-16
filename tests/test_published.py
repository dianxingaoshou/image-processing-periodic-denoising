"""Numerical contracts for the source-guided ARPENOS reg=0 port.

These tests verify numerical safety and qualitative removal, not equivalence to
MATLAB robustfit or replication of the authors' published benchmark numbers.
"""
import unittest

import numpy as np

from published_baseline import arpenos, denoise


class PublishedBaselineTests(unittest.TestCase):
    @staticmethod
    def sample(shape=(256, 256)):
        y, x = np.indices(shape)
        clean = (0.4 + 0.08 * np.sin(2 * np.pi * x / 256)
                 + 0.1 * np.exp(-((x - 90) ** 2 + (y - 140) ** 2) / (2 * 30 ** 2)))
        noisy = clean + 0.1 * np.sin(2 * np.pi * (0.147 * x + 0.093 * y) + 0.4)
        return clean, noisy

    def test_global_sinusoid_is_substantially_reduced(self):
        clean, noisy = self.sample()
        output, diagnostics = denoise(noisy, return_diagnostics=True)
        input_mse = np.mean((noisy - clean) ** 2)
        output_mse = np.mean((output - clean) ** 2)
        self.assertLess(output_mse, 0.25 * input_mse)
        self.assertEqual(diagnostics["patch_size"], 128)
        self.assertEqual(diagnostics["patch_count"], 64)
        self.assertGreater(diagnostics["outlier_fraction"], 0)
        self.assertTrue(np.isfinite(output).all())

    def test_constant_input_is_unchanged(self):
        for value in (0.0, 0.5, 1.0):
            constant = np.full((160, 192), value)
            np.testing.assert_array_equal(denoise(constant), constant)

    def test_odd_rectangular_images_and_conjugate_symmetry(self):
        _, noisy = self.sample((255, 251))
        output, diagnostics = denoise(noisy, return_diagnostics=True)
        self.assertEqual(output.shape, noisy.shape)
        self.assertTrue(np.isfinite(output).all())
        complex_residual = np.fft.ifft2(np.fft.fft2(noisy)
                                       * np.fft.ifftshift(diagnostics["notch_mask"]))
        self.assertLess(np.abs(complex_residual.imag).max(), 1e-12)

    def test_intensity_unit_change_is_equivariant(self):
        _, noisy = self.sample((160, 176))
        normalized = denoise(noisy)
        eight_bit_units = denoise(noisy * 255) / 255
        np.testing.assert_allclose(normalized, eight_bit_units, atol=1e-10, rtol=1e-10)

    def test_input_is_not_mutated(self):
        _, noisy = self.sample((160, 160))
        before = noisy.copy()
        first = denoise(noisy)
        second = denoise(noisy)
        np.testing.assert_array_equal(noisy, before)
        np.testing.assert_array_equal(first, second)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            arpenos(np.zeros((128, 128)), patch_size=128)
        with self.assertRaises(ValueError):
            arpenos(np.zeros((160, 160)), patch_size=63)
        with self.assertRaises(ValueError):
            arpenos(np.full((160, 160), np.nan))
        with self.assertRaises(ValueError):
            arpenos(np.zeros((160, 160, 3)))


if __name__ == "__main__":
    unittest.main()
