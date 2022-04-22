import unittest

import data
from data import SplitData


class TestData(unittest.TestCase):
    def test_shuffle_partition(self):
        inputs = [['i', 'want', 'candy'],
                  ['they', 're', 'red', 'hot'],
                  ['i', 'm', 'so', 'tired']]
        split_data = SplitData(inputs, [], [])
        chopped = split_data.shuffle_chop('train', 5)
        possibilities = ['012', '021', '102', '120', '201', '210']
        outcomes = [[inputs[int(ch)] + [data.END_OF_VERSE_TOKEN] for ch in poss] \
                    for poss in possibilities]
        flattened = [[el for lis in ell for el in lis] for ell in outcomes]
        padded = [el + [data.PAD_TOKEN] for el in flattened]
        padded = [[el[0:5], el[5:10], el[10:15]] for el in padded]
        matches = [int(el == chopped) for el in padded]
        n_matches = sum(matches)
        self.assertEqual(1, n_matches)

if __name__ == "__main__":
    unittest.main()
