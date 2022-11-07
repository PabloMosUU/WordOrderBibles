import unittest
import compression

class TestCompression(unittest.TestCase):
    def test_shortest_unseen_substrings(self):
        text = 'manzanas'
        expected = [1, 1, 1, 1, 3, 2, 2, 1]
        self.assertEqual(expected, compression.shortest_unseen_substrings(text))

if __name__ == "__main__":
    unittest.main()
