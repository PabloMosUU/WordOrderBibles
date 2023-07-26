import unittest

from analysis import unigram_entropy_direct, unigram_entropy_by_counts


class TestAnalysis(unittest.TestCase):
    def test_unigram_direct(self):
        text = 'The sky is overcast but the sky is nice'
        tokens = text.split(' ')
        self.assertAlmostEqual(1.82990, unigram_entropy_direct(tokens), places=5)

        lowercased = text.lower().split(' ')
        self.assertAlmostEqual(1.71879, unigram_entropy_direct(lowercased), places=5)

    def test_unigram_counts(self):
        text = 'the sky is overcast'
        tokens = text.split(' ')
        log_probas = {'the': -4.605, 'sky': -3.912, 'is': -3.507, 'overcast': -2.996, 'but': -2.652, 'nice': -2.207}
        self.assertAlmostEqual(5.417, unigram_entropy_by_counts(tokens, log_probas), places=3)


if __name__ == "__main__":
    unittest.main()
