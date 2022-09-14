import unittest

import numpy as np
import torch

import data
import train

class SimpleModel(train.LSTMLanguageModel):
    def __init__(self, embedding_dim, hidden_dim, n_layers: int, loss_function: torch.nn.Module,
                 avg_loss_per_token: bool, dropout: float, log_gradients: bool):
        # Take a simple language model
        # SOS -> the (80%), a (20%)
        # the, a -> dog (40%), cat (60%)
        # dog -> barked (90%), walked (10%)
        # cat -> meowed (70%), walked (20%), EOS (10%)
        # walked, barked, meowed -> EOS (100%)
        # EOS -> SOS (100%)
        self.all_words = [data.START_OF_VERSE_TOKEN, data.END_OF_VERSE_TOKEN, data.PAD_TOKEN] + \
                         "the a dog cat barked walked meowed".split()
        self.word_index = {word: i for i, word in enumerate(self.all_words)}
        super().__init__(embedding_dim, hidden_dim, self.word_index, n_layers, loss_function, avg_loss_per_token,
                         dropout, log_gradients)
        self.probs = {data.START_OF_VERSE_TOKEN: {'the': 0.8, 'a': 0.2},
                        'the': {'dog': 0.4, 'cat': 0.6},
                        'a': {'dog': 0.4, 'cat': 0.6},
                        'dog': {'barked': 0.9, 'walked': 0.1},
                        'cat': {'meowed': 0.7, 'walked': 0.2, data.END_OF_VERSE_TOKEN: 0.1},
                        'walked': {data.END_OF_VERSE_TOKEN: 1},
                        'barked': {data.END_OF_VERSE_TOKEN: 1},
                        'meowed': {data.END_OF_VERSE_TOKEN: 1},
                        data.END_OF_VERSE_TOKEN: {data.START_OF_VERSE_TOKEN: 1},
                      data.PAD_TOKEN: {}}
        self.index_word = train.invert_dict(self.word_index)


    def forward(self, batch_sequences, _):
        if len(batch_sequences) != 1:
            return torch.tensor([self.forward([seq], None)[0].numpy() for seq in batch_sequences])
        seq = batch_sequences[0]
        pred_word_scores = []
        for word in seq:
            scores = [-10000 for _ in range(len(self.all_words))]
            next_word_prob = self.probs[self.index_word[word.item()]]
            for next_word, prob in next_word_prob.items():
                scores[self.word_index[next_word]] = np.log(prob)
            pred_word_scores.append(scores)
        return torch.tensor([pred_word_scores])


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
        batch_tensor = train.truncate(full_dataset[1], is_input=True)
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
        batch_tensor = train.truncate(full_dataset[1], is_input=False)
        self.assertEqual(torch.Tensor, type(batch_tensor))
        self.assertEqual(word_index['line'], batch_tensor[0][0].item())


    def test_get_perplexity(self):
        test_words = [data.START_OF_VERSE_TOKEN] + "the dog walked".split() + [data.END_OF_VERSE_TOKEN]
        model = SimpleModel(300, 300, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
        test_sequence = torch.tensor([model.word_index[word] for word in test_words])
        # Probability: 0.8 * 0.4 * 0.1 * 1 = 0.032
        expected = 1.99054
        perplexity = model.get_perplexity(test_sequence)
        self.assertAlmostEqual(expected, perplexity, places=5)


    def test_get_perplexity_padded(self):
        model = SimpleModel(300, 300, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
        test_words = [data.START_OF_VERSE_TOKEN] + "the dog walked".split() + [data.END_OF_VERSE_TOKEN, data.PAD_TOKEN]
        test_sequence = torch.tensor([model.word_index[word] for word in test_words])
        # Probability: 0.8 * 0.4 * 0.1 * 1 = 0.032
        expected = 1.99054
        perplexity = model.get_perplexity(test_sequence)
        self.assertAlmostEqual(expected, perplexity, places=5)


    def test_perplexity(self):
        model = SimpleModel(300, 300, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
        test_words = [data.START_OF_VERSE_TOKEN] + "the dog walked".split() + [data.END_OF_VERSE_TOKEN]
        test_sequence = [[model.word_index[word] for word in test_words]]
        X = train.truncate(test_sequence, is_input=True)
        Y_true = train.truncate(test_sequence, is_input=False)
        Y_pred = model.forward(X, torch.tensor([len(test_words) - 1]))
        expected = 1.99054
        perplexity = model.perplexity(Y_true, Y_pred.permute(0, 2, 1), False)
        self.assertAlmostEqual(expected, perplexity, places=5)


    def test_perplexity_pad(self):
        model = SimpleModel(300, 300, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
        test_words = [data.START_OF_VERSE_TOKEN] + "the dog walked".split() + [data.END_OF_VERSE_TOKEN, data.PAD_TOKEN]
        test_sequence = [[model.word_index[word] for word in test_words]]
        X = train.truncate(test_sequence, is_input=True)
        Y_true = train.truncate(test_sequence, is_input=False)
        Y_pred = model.forward(X, torch.tensor([len(test_words) - 1]))
        expected = 1.99054
        perplexity = model.perplexity(Y_true, Y_pred.permute(0, 2, 1), False)
        self.assertAlmostEqual(expected, perplexity, places=5)


    def test_perplexity_two(self):
        model = SimpleModel(300, 300, 1, torch.nn.CrossEntropyLoss(), True, 0, False)
        test_sentences = ("the dog walked", "the cat")
        test_sequences = [data.to_indices(f'{data.START_OF_VERSE_TOKEN} {sent} {data.END_OF_VERSE_TOKEN}'.split(),
                                          model.word_index) \
                          for sent in test_sentences]
        # Pad the second sentence
        test_sequences[-1].append(model.word_index[data.PAD_TOKEN])
        X = train.truncate(test_sequences, is_input=True)
        Y_true = train.truncate(test_sequences, is_input=False)
        Y_pred = model.forward(X, torch.tensor([5, 4]))
        expected = 2.05411
        perplexity = model.perplexity(Y_true, Y_pred.permute(0, 2, 1), False)
        self.assertAlmostEqual(expected, perplexity, places=5)


if __name__ == "__main__":
    unittest.main()
