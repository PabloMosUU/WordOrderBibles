import unittest

from analysis import unigram_entropy_direct


class TestAnalysis(unittest.TestCase):
    def test_unigram_direct(self):
        text = 'The sky is overcast but the sky is nice'
        tokens = text.split(' ')
        self.assertAlmostEqual(1.82990, unigram_entropy_direct(tokens), places=5)

        lowercased = text.lower().split(' ')
        self.assertAlmostEqual(1.71879, unigram_entropy_direct(lowercased), places=5)


if __name__ == "__main__":
    unittest.main()
