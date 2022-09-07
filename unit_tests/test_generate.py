import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from torch import tensor

import data
import generate


class TestGenerate(unittest.TestCase):
    def test_beam_search_decoder(self):
        # noinspection PyUnusedLocal
        def mock_get_next_word_log_probabilities(lm, sequence, word_index, index_word):
            # define a sequence of 10 words over a vocab of 5 words
            mock_data = [[0.1, 0.2, 0.3, 0.4, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1],
                         [0.1, 0.2, 0.3, 0.4, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1],
                         [0.1, 0.2, 0.3, 0.4, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1],
                         [0.1, 0.2, 0.3, 0.4, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1],
                         [0.1, 0.2, 0.3, 0.4, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1]]
            row = mock_data[len(sequence)-1]
            return [(ii, np.log(el)) for ii, el in enumerate(row)]

        model = MagicMock()
        seed = [data.START_OF_VERSE_TOKEN]
        k = 3
        length = 10
        with patch('generate._get_next_word_log_probabilities', side_effect=mock_get_next_word_log_probabilities):
            generated_words = generate.beam_search_decoder(model, seed, k, length)
        expected = [[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0], 6.931471805599453],
                    [[4, 0, 4, 0, 4, 0, 4, 0, 4, 1], 7.154615356913663],
                    [[4, 0, 4, 0, 4, 0, 4, 0, 3, 0], 7.154615356913663]]
        for i, expected_candidate in enumerate(expected):
            self.assertEqual(expected_candidate[0], generated_words[i][0][-length:])
            self.assertEqual(expected_candidate[1], generated_words[i][1])

    def test_get_next_word_log_probabilities(self):

        def almost_equal(a, b):
            return abs(a-b) / (a+b) < 0.001

        token_scores = tensor([[[2.0133, 6.5948, 3.9017, 2.2569, 2.2569, 1.3556, 0.0311, 0.6919],
                                [1.6830, 5.7962, 12.0348, 5.2092, 2.2082, 3.5261, -4.3216, 0.1037],
                                [0.3161, 2.8529, 5.0923, 13.0054, 6.2734, 1.4280, -2.3865, 2.2714]]])
        model = MagicMock(return_value=token_scores)
        seq = ['this', 'd', 'f']
        word_ix = {'this': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
        ix_word = {v: k for k, v in word_ix.items()}
        log_probabilities = generate._get_next_word_log_probabilities(model, seq, word_ix, ix_word)
        # Test that the model was called as expected
        model.assert_called_once()
        actual_call_args = model.call_args.args
        self.assertEqual(1, len(actual_call_args[0]))
        self.assertEqual([0, 4, 6], list(actual_call_args[0][0]))
        self.assertEqual(3, actual_call_args[1].item())
        # Test that the returned probabilities make sense
        expected = [('this', -12.690930171377245), ('a', -10.154130171377245), ('b', -7.914730171377245),
                    ('c', -0.0016301713772450066), ('d', -6.7336301713772455), ('e', -11.579030171377244),
                    ('f', -15.393530171377245), ('g', -10.735630171377245)]
        self.assertTrue(all([expected[i][0] == lp[0] and almost_equal(expected[i][1], lp[1]) \
                             for i, lp in enumerate(log_probabilities)]))


if __name__ == "__main__":
    unittest.main()
