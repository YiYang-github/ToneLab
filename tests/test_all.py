"""
test_all.py
Unit test module.
"""

import sys
import time
import unittest
sys.path.append("")
import numpy as np
from tonelab.tone2vec import pitch_curve, curve_simi, tone_feats


class TestTone2Vec(unittest.TestCase):
    """
    Unit test module for tone2vec.py
    """

    def test_pitch_curve_length_2(self):
        seq = [2.0, 4.0]
        curve = pitch_curve(seq)
        x = np.array([1, 2, 3])
        expected_y = np.array([2.0, 3.0, 4.0])  # Linear interpolation
        np.testing.assert_almost_equal(curve(x), expected_y)

    def test_pitch_curve_length_3(self):
        seq = [1.0, 3.0, 5.0]
        curve = pitch_curve(seq)
        x = np.array([1, 2, 3])
        expected_y = np.array([1.0, 3.0, 5.0])  # Quadratic interpolation
        np.testing.assert_almost_equal(curve(x), expected_y)

    def test_pitch_curve_invalid_length(self):
        seq = [1.0]  # Length not 2 or 3
        with self.assertRaises(ValueError):
            pitch_curve(seq)

    def test_curve_simi_identical_curves(self):
        seq1 = [1.0, 3.0, 5.0]
        seq2 = [1.0, 3.0, 5.0]
        area = curve_simi(seq1, seq2)
        self.assertEqual(area, 0.0)

    def test_curve_simi_different_curves(self):
        seq1 = [1.0, 3.0, 5.0]
        seq2 = [2.0, 4.0, 6.0]
        area = curve_simi(seq1, seq2)
        self.assertGreater(area, 0.0)  # Ensure the area is positive

    def test_curve_simi_length_2_curves(self):
        seq1 = [1.0, 3.0]
        seq2 = [2.0, 4.0]
        area = curve_simi(seq1, seq2)
        self.assertGreater(area, 0.0)  # Ensure the area is positive

    def test_tone_feats(self):
        tone_list = [[[1, 2, 3], [2, 3, 4]], [[0, 1, 2], [3, 4, 5]]]  # Example indices for tones
        features = tone_feats(tone_list, 'PCA', 2)
        features = tone_feats(tone_list, 't-SNE', 2)
        features = tone_feats(tone_list, 'UMAP', 2)

    

if __name__ == '__main__':
    unittest.main()

