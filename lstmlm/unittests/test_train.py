import unittest

import torch

import lstmlm.train as train

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

if __name__ == "__main__":
    unittest.main()
