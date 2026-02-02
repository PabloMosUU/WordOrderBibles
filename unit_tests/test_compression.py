import sys
import os
# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

import unittest
import compression

class TestCompression(unittest.TestCase):
    def test_shortest_unseen_substrings(self):
        text = 'manzanas'
        expected = [1, 1, 1, 1, 3, 2, 2, 1]
        self.assertEqual(expected, compression.shortest_unseen_subsequence_lengths(text))

    def test_shortest_unseen_substrings_list(self):
        text = ['m', 'a', 'n', 'z', 'a', 'n', 'a', 's']
        expected = [1, 1, 1, 1, 3, 2, 2, 1]
        self.assertEqual(expected, compression.shortest_unseen_subsequence_lengths(text))

    def test_shortest_unseen_substrings_number_list(self):
        text = [0, 1, 2, 3, 1, 2, 1, 4]
        expected = [1, 1, 1, 1, 3, 2, 2, 1]
        self.assertEqual(expected, compression.shortest_unseen_subsequence_lengths(text))

if __name__ == "__main__":
    unittest.main()
