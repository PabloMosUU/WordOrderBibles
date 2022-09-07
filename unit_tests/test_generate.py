import unittest
from unittest.mock import patch

from torch import nn

import data
import generate
import numpy as np

class TestGenerate(unittest.TestCase):
    def test_beam_search_decoder(self):
        def mock_get_next_word_log_probabilities(_, sequence):
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
            return [(i, np.log(el)) for i, el in enumerate(row)]

        model = nn.Module()
        seed = [data.START_OF_VERSE_TOKEN]
        k = 3
        length = 10
        with patch('generate._get_next_word_log_probabilities', side_effect=mock_get_next_word_log_probabilities):
            generated_words = generate.beam_search_decoder(model, seed, k, length)
        self.assertEqual([4, 0, 4, 0, 4, 0, 4, 0, 4, 0], generated_words[0][-length:])
        self.assertEqual(6.931471805599453, generated_words[1])


if __name__ == "__main__":
    unittest.main()
