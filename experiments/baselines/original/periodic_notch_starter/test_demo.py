"""Run with: python -m unittest -v test_demo.py"""
import unittest
import numpy as np
from demo import Config, conjugate_position, denoise, synthesize

class NotchTests(unittest.TestCase):
    def test_constant_image_is_unchanged(self):
        a = np.full((65, 79), 0.5)
        b, peaks, mask = denoise(a)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(peaks, [])
        np.testing.assert_array_equal(mask, np.ones_like(a))

    def test_odd_rectangular_image_and_symmetry(self):
        h, w = 129, 151
        y, x = np.indices((h, w))
        clean = .3 + .08 * x / w + .08 * y / h
        noisy, _ = synthesize(clean, [(23., 37., .12, .7)])
        restored, peaks, mask = denoise(noisy)
        self.assertGreaterEqual(len(peaks), 1)
        self.assertLess(np.mean((clean-restored)**2), .1*np.mean((clean-noisy)**2))
        yr = (2*(h//2)-np.arange(h)) % h
        xr = (2*(w//2)-np.arange(w)) % w
        np.testing.assert_allclose(mask, mask[np.ix_(yr, xr)], atol=1e-12)
        self.assertEqual(mask[h//2, w//2], 1.)

    def test_even_image_pair_map(self):
        self.assertEqual(conjugate_position(12, 9, (64, 80)), (52, 71))
        self.assertEqual(conjugate_position(32, 40, (64, 80)), (32, 40))

    def test_invalid_inputs(self):
        for image in [np.zeros((8, 8)), np.ones((32, 32))*2, np.ones((32, 32))*np.nan]:
            with self.assertRaises(ValueError):
                denoise(image)
        with self.assertRaises(ValueError):
            Config(strength=2)

if __name__ == '__main__':
    unittest.main()
