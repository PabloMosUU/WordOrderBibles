import json
import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
