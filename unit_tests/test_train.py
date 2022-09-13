import unittest
from unittest import mock

import numpy as np
import torch

import data
import train

class TestTrain(unittest.TestCase):
    def test_select_batch_input(self):
        full_dataset = [el.split() for el in \
                        ['I love it <PAD>', 'My name is Earl',
                         'Whose line is it anyway', 'As you like it <PAD>',
                         'Everybody']]
        words = list(set([el for lis in full_dataset for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        full_dataset = [[word_index[word] for word in sequence] for sequence in full_dataset]
        full_dataset = [full_dataset[0:2], full_dataset[2:4], full_dataset[4:5]]
        batch_tensor = train.select_batch(full_dataset, batch_ix=1, is_input=True)
        self.assertEqual(torch.Tensor, type(batch_tensor))
        self.assertEqual(word_index['Whose'], batch_tensor[0][0].item())


    def test_select_batch_target(self):
        full_dataset = [el.split() for el in \
                        ['I love it <PAD>', 'My name is Earl',
                         'Whose line is it anyway', 'As you like it <PAD>',
                         'Everybody']]
        words = list(set([el for lis in full_dataset for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        full_dataset = [[word_index[word] for word in sequence] for sequence in full_dataset]
        full_dataset = [full_dataset[0:2], full_dataset[2:4], full_dataset[4:5]]
        batch_tensor = train.select_batch(full_dataset, batch_ix=1, is_input=False)
        self.assertEqual(torch.Tensor, type(batch_tensor))
        self.assertEqual(word_index['line'], batch_tensor[0][0].item())


    def test_get_perplexity(self):
        # Take a simple language model
        # SOS -> the (80%), a (20%)
        # the, a -> dog (40%), cat (60%)
        # dog -> barked (90%), walked (10%)
        # cat -> meowed (80%), walked (20%)
        # walked, barked, meowed -> EOS (100%)
        # EOS -> SOS (100%)
        all_words = [data.START_OF_VERSE_TOKEN, data.END_OF_VERSE_TOKEN, data.PAD_TOKEN] + \
                    "the a dog cat barked walked meowed".split()
        word_index = {word: i for i, word in enumerate(all_words)}
        probs = {data.START_OF_VERSE_TOKEN: {'the': 0.8, 'a': 0.2},
                 'the': {'dog': 0.4, 'cat': 0.6},
                 'a': {'dog': 0.4, 'cat': 0.6},
                 'dog': {'barked': 0.9, 'walked': 0.1},
                 'cat': {'meowed': 0.8, 'walked': 0.2},
                 'walked': {data.END_OF_VERSE_TOKEN: 1},
                 'barked': {data.END_OF_VERSE_TOKEN: 1},
                 'meowed': {data.END_OF_VERSE_TOKEN: 1},
                 data.END_OF_VERSE_TOKEN: {data.START_OF_VERSE_TOKEN: 1}}
        index_word = train.invert_dict(word_index)

        # noinspection PyUnusedLocal
        def mock_forward(_, batch_sequences, lengths):
            if len(batch_sequences) != 1:
                raise ValueError('Only implemented for batches of size 1')
            seq = batch_sequences[0]
            pred_word_scores = []
            for word in seq:
                scores = [-10000 for _ in range(len(all_words))]
                next_word_prob = probs[index_word[word.item()]]
                for next_word, prob in next_word_prob.items():
                    scores[word_index[next_word]] = np.log(prob)
                pred_word_scores.append(scores)
            return torch.tensor([pred_word_scores])

        test_sequence = [data.START_OF_VERSE_TOKEN] + "the dog walked".split() + [data.END_OF_VERSE_TOKEN]
        # Probability: 0.8 * 0.4 * 0.1 * 1 = 0.032
        expected = 1.99054
        # Mock the forward method
        with mock.patch.object(train.LSTMLanguageModel, 'forward', new=mock_forward):
            model = train.LSTMLanguageModel(300, 300, word_index, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
            perplexity = model.get_perplexity(test_sequence)
        self.assertAlmostEqual(expected, perplexity, places=5)


if __name__ == "__main__":
    unittest.main()
