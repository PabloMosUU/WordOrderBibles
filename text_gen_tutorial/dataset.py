import torch
import pandas as pd
from collections import Counter

class TrainArgs:
    def __init__(self, max_epochs, batch_size, sequence_length):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __repr__(self):
        return f'Max epochs: {self.max_epochs}, Batch size: {self.batch_size}, Sequence length: {self.sequence_length}'

# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: TrainArgs,
    ):
        self.args = args
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()

        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    @staticmethod
    def load_words():
        train_df = pd.read_csv('data/reddit-cleanjokes.csv')
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')

    def get_unique_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )
