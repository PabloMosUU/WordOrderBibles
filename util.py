import json
import math

import pandas as pd
from matplotlib import pyplot as plt
import data
import torch.nn as nn
import torch
import numpy as np

from data import batch
import configparser

BOOK_ID_NAME = {'40': 'Matthew',
                '41': 'Mark',
                '42': 'Luke',
                '43': 'John',
                '44': 'Acts',
                '66': 'Revelation'}


class Token:
    def __init__(self, token: str, is_start_of_word: bool):
        self.token = token
        self.is_start_of_word = is_start_of_word

    def __repr__(self):
        return ('' if self.is_start_of_word else '#') + self.token

    def __eq__(self, other):
        return isinstance(other, Token) and self.token == other.token and \
            self.is_start_of_word == other.is_start_of_word


class TrainConfig:
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            n_layers: int,
            learning_rate: float,
            n_epochs: int,
            clip_gradients: bool,
            optimizer: str,
            weight_decay: float,
            batch_size: int,
            dropout: float,
            verbose: bool,
            gradient_logging: bool,
            avg_loss_per_token: bool,
            validation_metrics: list
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.clip_gradients = clip_gradients
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.verbose = verbose
        self.gradient_logging = gradient_logging
        self.avg_loss_per_token = avg_loss_per_token
        self.validation_metrics = validation_metrics

    def __repr__(self):
        return ', '.join([f'{k}: {v}' for k, v in self.to_dict().items()])

    def to_dict(self):
        return {'embedding_dim': self.embedding_dim, 'hidden_dim': self.hidden_dim, 'n_layers': self.n_layers,
                'learning_rate': self.learning_rate, 'n_epochs': self.n_epochs, 'clip_gradients': self.clip_gradients,
                'optimizer': self.optimizer, 'weight_decay': self.weight_decay, 'batch_size': self.batch_size,
                'dropout': self.dropout, 'verbose': self.verbose, 'gradient_logging': self.gradient_logging,
                'avg_loss_per_token': self.avg_loss_per_token, 'validation_metrics': ' '.join(self.validation_metrics)}

    def save(self, filename):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {k: str(v) for k, v in self.to_dict().items()}
        with open(filename, 'w') as f:
            config.write(f)


class LSTMLanguageModel(nn.Module):

    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            word_index: dict,
            n_layers: int,
            loss_function: nn.Module,
            avg_loss_per_token: bool,
            dropout: float,
            log_gradients: bool
    ):
        super(LSTMLanguageModel, self).__init__()
        self.word_index = word_index
        vocab_size = len(self.word_index)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.word_index[data.PAD_TOKEN])

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.loss_function = loss_function
        self.avg_loss_per_token = avg_loss_per_token
        self.perplexity_loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=word_index[data.PAD_TOKEN],
            reduction='sum'
        )

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        # Variables to store the number of training and validation data points and the number of epochs
        self.train_size = None
        self.validation_size = None
        self.n_epochs = None
        self.gradient_logging = log_gradients
        self.big_gradients = []
        self.epoch = -1

    def forward(self, sequences, original_sequence_lengths: torch.Tensor):
        embeds = self.word_embeddings(sequences)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeds, original_sequence_lengths, batch_first=True)

        lstm_out, _ = self.lstm(packed_embeddings)

        # undo the packing operation
        padded_lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        return self.hidden2word(padded_lstm_outputs)

    def loss(self, y_true, y_pred):
        return self.loss_function(y_pred, y_true)

    def perplexity(self, y_true: torch.Tensor, y_pred: torch.Tensor, concatenate: bool) -> float:
        with torch.no_grad():
            if concatenate:
                raise NotImplementedError()
            total_loss = self.perplexity_loss_function(y_pred, y_true).item()
            # The additional len(Y_true) accounts for START_OF_VERSE_TOKEN
            n_tokens = torch.sum(y_true != self.word_index[data.PAD_TOKEN]).item() + len(y_true)
            return np.exp(total_loss / n_tokens)

    def save(self, filename: str) -> None:
        torch.save(self, filename)
        if self.gradient_logging:
            with open(f'{filename}.log', 'w') as logfile:
                logfile.write('\n'.join([', '.join([str(el) for el in t]) for t in self.big_gradients]))

    @staticmethod
    def load(filename: str):
        return torch.load(filename)

    def log_gradients(self, batch_ix: int):
        if self.gradient_logging:
            for k, v in {'embed': self.word_embeddings, 'lstm': self.lstm, 'hidden2word': self.hidden2word}.items():
                for i, p in enumerate(v.parameters()):
                    gradients = abs(p.grad.flatten())
                    if any(gradients > 0.5):
                        self.big_gradients.append(
                            (self.epoch, batch_ix, k, i, np.array2string(gradients.detach().numpy()))
                        )

    def get_perplexity(self, corpus: list, concatenate: bool) -> float:
        """
        Compute the perplexity for an entire corpus
        :param corpus: a corpus represented as a list of sequences, each of which is a list of tokens
        :param concatenate: whether we want to concatenate the entire corpus together for perplexity computations
        :return: the perplexity on the entire corpus
        """
        dataset, original_sequence_lengths = batch(corpus, len(corpus), self.word_index)
        x = truncate(dataset[0], True)
        y = truncate(dataset[0], False)
        y_pred = self.forward(x, torch.tensor([seq_len - 1 for seq_len in original_sequence_lengths[0]]))
        return self.perplexity(y, y_pred.permute(0, 2, 1), concatenate)


def truncate(selected_batch: list, is_input: bool) -> torch.Tensor:
    """
    Select the indexed batch from the dataset, which is assumed to be padded
    :param selected_batch: the batch that we want to truncate and turn into a tensor
    :param is_input: whether we want to process these sequences as inputs (as opposed to targets)
    :return: the tensor with the adjusted sequences
    """
    # Convert to inputs or targets
    truncated = [seq[:len(seq)-1] if is_input else seq[1:] for seq in selected_batch]

    # Convert to a PyTorch tensor
    return torch.tensor(truncated)


def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v: k for k, v in key_val.items()}


def log_factorial(x: int) -> float:
    return math.lgamma(x) + np.log(x)


def to_csv(json_file: str) -> None:
    # Read the JSON file
    with open(json_file, 'r') as f:
        book_entropies = json.loads(f.read())
    # Parse the dictionaries into a list of rows, each of which is a dictionary, all with the same keys    rows = []
    row_list = []
    for book_id, version_entropies in book_entropies.items():
        for n_iter, entropies_types in version_entropies.items():
            level_entropies = entropies_types
            csv_row = level_entropies.copy()
            csv_row['book_id'] = book_id
            csv_row['iter_id'] = n_iter
            row_list.append(csv_row)
    if not row_list:
        empty_df = pd.DataFrame(columns="orig,shuffled,masked,book_id,iter_id,book,D_structure,D_order".split(','))
        empty_df.to_csv(json_file.replace('.json', '.csv'), index=False)
        return
    # Create a Pandas dataframe
    df = pd.DataFrame(row_list)
    # Perform a check
    assert_valid(df)
    # Map book IDs to their names
    df['book'] = df['book_id'].map(BOOK_ID_NAME)
    # Compute the quantities that are plotted by Koplenig et al.
    df['D_structure'] = df.apply(lambda row: row['masked'] - row['orig'], 1)
    df['D_order'] = df.apply(lambda row: row['shuffled'] - row['orig'], 1)
    df.to_csv(json_file.replace('.json', '.csv'), index=False)
    return


def assert_valid(df: pd.DataFrame) -> None:
    for book_id in df.book_id.unique():
        for iter_id in df.iter_id.unique():
            selection = df[df.apply(lambda row: row['book_id'] == book_id and row['iter_id'] == iter_id,
                                    1)]
            if len(selection) == 0:
                continue
            if len(selection) != 1:
                assert len(selection) == 2 and iter_id in ('0', '1000')
                for col in ('orig', 'shuffled', 'masked'):
                    assert rel_error(selection[col].tolist()) * 100 < 0.5
    return


def rel_error(a):
    assert len(a) == 2
    return abs(a[0] - a[1]) / (a[0] + a[1])


def make_book_plot(grp: pd.DataFrame, book: str, bible: str) -> tuple:
    """
    Make a correlation plot of word-order and word-structure information for a specific bible and book
    :param grp: a dataframe containing only the single bible for which you want to make the plot
    :param book: the book that you wish to select
    :param bible: the name of the bible, for printing purposes
    :return: figure and axes for the plot
    """
    df = grp[grp['book'] == book]
    # Return ax instead of saving, so don't take an output dir as a parameter
    xs = df[df['experiment'] == 'splitting']['D_order'].tolist()
    ys = df[df['experiment'] == 'splitting']['D_structure'].tolist()
    xp = df[df['experiment'] == 'pasting']['D_order'].tolist()
    yp = df[df['experiment'] == 'pasting']['D_structure'].tolist()
    labels_splitting = df[df['experiment'] == 'splitting']['iter_id'].tolist()
    labels_pasting = df[df['experiment'] == 'pasting']['iter_id'].tolist()
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, label='splitting')
    ax.scatter(xp, yp, label='pasting')
    plt.xlabel('Word order information')
    plt.ylabel('Word structure information')
    plt.title(f'{book} ({bible})')
    plt.legend()
    for i, txt in enumerate(labels_splitting):
        ax.annotate(txt, (xs[i], ys[i]), rotation=45)
    for i, txt in enumerate(labels_pasting):
        ax.annotate(txt, (xp[i], yp[i]), rotation=45)
    return fig, ax


def make_plot_(df: pd.DataFrame, bible_filename: str, output_dir: str) -> None:
    for lbl, grp in df.groupby('book'):
        fig, ax = make_book_plot(grp, str(lbl), bible_filename)
        if output_dir != '':
            fig.savefig(f"{output_dir}/{bible_filename.replace('.txt', '')}_{lbl}.png")
