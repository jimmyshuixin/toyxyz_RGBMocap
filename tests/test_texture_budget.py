import unittest
import numpy as np

from mocap.retarget.texture_budget import allocate_preview_sizes, area_reduce


class TextureBudgetTests(unittest.TestCase):
    def test_fitting_images_retain_original_dimensions(self):
        original = {'a': [17, 13], 'b': [32, 20]}
        sizes, factors = allocate_preview_sizes(original)
        self.assertEqual(sizes, original)
        self.assertEqual(factors, {'a': 1, 'b': 1})

    def test_largest_first_fits_shared_and_single_image_limits(self):
        original = {'body': [8192, 8192], 'stocking': [8192, 4096],
                    'accessory': [4096, 4096], 'face': [2048, 2048]}
        sizes, factors = allocate_preview_sizes(original, total_bytes=400*1024**2)
        self.assertEqual(sizes['body'], [4096, 4096])
        self.assertEqual(factors, {'body': 2, 'stocking': 1, 'accessory': 1, 'face': 1})
        sizes, factors = allocate_preview_sizes({'large': [16384, 16384]})
        self.assertEqual(sizes, {'large': [8192, 8192]})

    def test_odd_dimensions_preserve_boundary_texels(self):
        sizes, factors = allocate_preview_sizes({'a': [5, 3]}, image_bytes=24, total_bytes=24)
        self.assertEqual((sizes, factors), ({'a': [3, 2]}, {'a': 2}))
        pixels = np.arange(15, dtype=np.float32).reshape(3, 5, 1)
        np.testing.assert_array_equal(area_reduce(pixels, 2)[..., 0], [[3., 5., 6.5], [10.5, 12.5, 14.]])

    def test_uniform_color_and_alpha_survive_reduction(self):
        pixels = np.broadcast_to(np.array([.2, .4, .6, 0.], np.float32), (16, 16, 4))
        np.testing.assert_allclose(area_reduce(pixels, 4), np.broadcast_to(pixels[0, 0], (4, 4, 4)), atol=1e-7)

    def test_impossible_minimum_budget_and_bad_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            allocate_preview_sizes({'a': [1, 1], 'b': [1, 1]}, total_bytes=4)
        for size in ([0, 1], [-1, 2], [3.5, 2]):
            with self.assertRaises(ValueError):
                allocate_preview_sizes({'a': size})


if __name__ == '__main__':
    unittest.main()
