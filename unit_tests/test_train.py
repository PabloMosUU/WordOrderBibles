import unittest

import torch

import data
import train
from train import batch, pad_batch


class TestData(unittest.TestCase):
    def test_batch(self):
        sequences = [el.split() for el in \
                     ['I love it', 'My name is Earl', 'Whose line is it anyway', 'As you like it', 'Everybody']]
        words = list(set([el for lis in sequences for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        word_index[data.PAD_TOKEN] = 170
        word_index[data.START_OF_VERSE_TOKEN] = 171
        word_index[data.END_OF_VERSE_TOKEN] = 172
        dataset, original_sequence_lengths = batch(sequences, 2, word_index)
        self.assertEqual(3, len(dataset))
        self.assertTrue(all([len(el) <= 2 for el in dataset]))
        self.assertTrue(all([len(seq) == 6 for seq in dataset[0]]))
        self.assertTrue(all([len(seq) == 7 for seq in dataset[1]]))
        self.assertTrue(all([all([seq[0] == 171 for seq in b]) for b in dataset]))
        self.assertTrue(all([all([seq[-1] in (170, 172) for seq in b]) for b in dataset]))
        self.assertEqual(172, dataset[0][1][-2])
        self.assertEqual(6, original_sequence_lengths[1][1])

    def test_batch_sorted(self):
        sequences = [el.split() for el in \
                     ['I love it', 'My name is Earl', 'Whose line is it anyway', 'As you like it', 'Everybody']]
        words = list(set([el for lis in sequences for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        index_word = {ix: wd for wd, ix in word_index.items()}
        word_index[data.PAD_TOKEN] = 170
        word_index[data.START_OF_VERSE_TOKEN] = 171
        word_index[data.END_OF_VERSE_TOKEN] = 172
        dataset, _ = batch(sequences, 2, word_index)
        self.assertEqual('My', index_word[dataset[0][0][1]])

    def test_pad_batch(self):
        sequences = [el.split() for el in ['I love it', 'My name is Earl']]
        padded = pad_batch(sequences)
        self.assertEqual(2, len(padded))
        self.assertTrue(all([len(el) == 4 for el in padded]))
        self.assertEqual(data.PAD_TOKEN, padded[0][-1])

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

if __name__ == "__main__":
    unittest.main()
