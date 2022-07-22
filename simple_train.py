"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
import torch.nn as nn
import torch
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt


class LSTMLanguageModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMLanguageModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # TODO: allow choosing the number of layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        next_word_space = self.hidden2word(lstm_out.view(len(sequence), -1))
        next_word_scores = functional.log_softmax(next_word_space, dim=1)
        return next_word_scores

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    @staticmethod
    def load(filename: str) -> nn.Module:
        return torch.load(filename)

class TrainConfig:
    def __init__(self, embedding_dim: int, hidden_dim: int, n_layers: int, learning_rate: float, n_epochs: int):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs


def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}

def get_next_words(scores: torch.Tensor, ix_next_word: dict) -> np.ndarray:
    pred_ixs = scores.max(dim=1).indices.numpy()
    return np.vectorize(lambda ix: ix_next_word[ix])(pred_ixs)

def get_word_index(sequences: list) -> dict:
    """
    Generate a look up word->index dictionary from a corpus
    :param sequences: a list of lists of tokens
    :return: a map from word to index
    """
    word_ix = {}
    # For each words-list (sentence) in the training_data
    for sent in sequences:
        for word in sent:
            if word not in word_ix:  # word has not been assigned an index yet
                word_ix[word] = len(word_ix)  # Assign each word with a unique index
    word_ix[data.UNKNOWN_TOKEN] = len(word_ix)
    word_ix[data.CHUNK_END_TOKEN] = len(word_ix)
    return word_ix

def train_sample_(model: nn.Module, sample: list, word_ix: dict, loss_function, optimizer) -> float:
    # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
    model.zero_grad()

    # Step 2. Get our inputs ready for the network, that is, turn them into tensors of word indices.
    sentence_in = prepare_sequence([data.START_OF_VERSE_TOKEN] + sample, word_ix)
    targets = prepare_sequence(sample + [data.END_OF_VERSE_TOKEN], word_ix)

    # Step 3. Run our forward pass.
    partial_pred_scores = model(sentence_in)

    # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
    loss = loss_function(partial_pred_scores, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_sample_(model: nn.Module, sample: list, word_ix: dict, loss_function: nn.Module) -> float:
    # Get our inputs ready for the network, that is, turn them into tensors of word indices.
    sentence_in = prepare_sequence([data.START_OF_VERSE_TOKEN] + sample, word_ix)
    targets = prepare_sequence(sample + [data.END_OF_VERSE_TOKEN], word_ix)

    # Run our forward pass.
    partial_pred_scores = model(sentence_in)

    # Compute the loss
    loss = loss_function(partial_pred_scores, targets)

    return loss.item()

def train_(model: nn.Module,
           corpus: list,
           word_ix: dict,
           n_epochs: int,
           loss_function,
           optimizer,
           verbose=False,
           validate=False,
           validation_set=None
           ) -> tuple:
    if validation_set is None:
        validation_set = []
    epoch_train_loss, epoch_val_loss = [], []
    for epoch in range(n_epochs):
        if verbose and (int(n_epochs/10) == 0 or epoch % int(n_epochs/10) == 0):
            print(f'INFO: processing epoch {epoch}')
        epoch_loss = 0
        for i, training_sentence in enumerate(corpus):
            if verbose and i % int(len(corpus)/10) == 0:
                print(f'\tINFO: processing sentence {i}')
            epoch_loss += train_sample_(model, training_sentence, word_ix, loss_function, optimizer)

        if validate:
            epoch_val_loss.append(validate_(model, validation_set, word_ix, loss_function))

        epoch_train_loss.append(epoch_loss / len(corpus))
    return epoch_train_loss, epoch_val_loss

def validate_(model: nn.Module, validation_set: list, word_ix: dict, loss_function: nn.Module) -> float:
    loss = 0
    with torch.no_grad():
        for validation_sentence in validation_set:
            loss += validate_sample_(model, validation_sentence, word_ix, loss_function)
    return loss / len(validation_set)

def pred_sample(model: nn.Module, sample: list, word_ix: dict, ix_word: dict) -> np.ndarray:
    words = sample.copy()
    for i in range(1, len(sample)):
        seq = prepare_sequence(words, word_ix)
        trained_next_word_scores = model(seq)
        word_i = get_next_words(trained_next_word_scores, ix_word)[i-1]
        words[i] = word_i
    return np.array(words)

def pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> list:
    with torch.no_grad():
        return [pred_sample(model, seq, word_ix, ix_word) for seq in corpus]

def print_pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> None:
    predictions = pred(model, corpus, word_ix, ix_word)
    for prediction in predictions:
        print(' '.join(prediction))

def initialize_model(embedding_dim, hidden_dim, words_dim, lr):
    model = LSTMLanguageModel(embedding_dim, hidden_dim, words_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, loss_function, optimizer

def plot_losses(loss_by_epoch: list) -> None:
    assert len(set([len(dataset_losses) for dataset_losses in loss_by_epoch])) == 1
    for dataset_losses in loss_by_epoch:
        plt.plot(range(len(dataset_losses)), dataset_losses)
    plt.show()

def save_losses(dataset_epoch_losses: dict, filename: str) -> None:
    """
    Save the loss versus epoch for each dataset
    :param dataset_epoch_losses: a map from dataset name (train, val) to a list of losses per epoch
    :param filename: the name of the file where the losses should be saved
    :return: nothing. The losses are saved to a file
    """
    with open(filename, 'w') as f:
        for k, v in dataset_epoch_losses.items():
            f.write(k + '\n')
            f.write(', '.join([str(el) for el in v]) + '\n')

def load_losses(filename: str) -> dict:
    with open(filename, 'r') as f:
        lines = f.readlines()
    dataset_losses = {}
    for i in range(int(len(lines) / 2)):
        dataset_losses[lines[2*i].strip()] = [float(el.strip()) for el in lines[2*i+1].split(',')]
    return dataset_losses

def get_perplexity(loss: float) -> float:
    """
    Computes the perplexity given the loss. It is equivalent to torch.exp(loss_tensor).item()
    :param loss: the loss computed from a language model
    :return: the perplexity of the language model
    """
    return np.exp(loss)


def prepare_sequence(seq: list, to_ix: dict) -> torch.Tensor:
    index_sequence = [to_ix[w] if w in to_ix else to_ix[data.UNKNOWN_TOKEN] for w in seq]
    return torch.tensor(index_sequence, dtype=torch.long)


def to_train_config(cfg: configparser.ConfigParser, version: str) -> TrainConfig:
    params = cfg[version]
    return TrainConfig(
        int(params['embedding_dim']),
        int(params['hidden_dim']),
        int(params['n_layers']),
        float(params['learning_rate']),
        int(params['n_epochs'])
    )
