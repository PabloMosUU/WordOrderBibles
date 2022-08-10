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

from data import prepare_sequence
import sys

class LSTMLanguageModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_index: dict, n_layers: int, loss_function: nn.Module):
        super(LSTMLanguageModel, self).__init__()
        self.word_index = word_index
        vocab_size = len(self.word_index)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.word_index[data.PAD_TOKEN])

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.loss_function = loss_function

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        # Variables to store the number of training and validation data points and the number of epochs
        self.train_size = None
        self.validation_size = None
        self.n_epochs = None

    def forward(self, sequences, original_sequence_lengths: torch.Tensor):
        embeds = self.word_embeddings(sequences)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeds, original_sequence_lengths, batch_first=True)

        lstm_out, _ = self.lstm(packed_embeddings)

        # undo the packing operation
        padded_lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        next_word_space = self.hidden2word(padded_lstm_outputs)
        next_word_scores = functional.log_softmax(next_word_space, dim=1)
        return next_word_scores

    def loss(self, Y_true, Y_pred):
        return self.loss_function(Y_pred, Y_true)

    def save(self, filename: str) -> None:
        torch.save(self, filename)

    @staticmethod
    def load(filename: str) -> nn.Module:
        return torch.load(filename)

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
            batch_size: int
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.clip_gradients = clip_gradients
        self.optimizer = optimizer
        self.batch_size = batch_size

    def __repr__(self):
        return ', '.join([f'{k}: {v}' for k, v in self.to_dict().items()])

    def to_dict(self):
        return {'embedding_dim': self.embedding_dim, 'hidden_dim': self.hidden_dim, 'n_layers': self.n_layers,
                'learning_rate': self.learning_rate, 'n_epochs': self.n_epochs, 'clip_gradients': self.clip_gradients,
                'optimizer': self.optimizer, 'batch_size': self.batch_size}

    def save(self, filename):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {k: str(v) for k, v in self.to_dict().items()}
        with open(filename, 'w') as f:
            config.write(f)


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
    special_tokens = (
        data.UNKNOWN_TOKEN,
        data.CHUNK_END_TOKEN,
        data.START_OF_VERSE_TOKEN,
        data.END_OF_VERSE_TOKEN,
        data.PAD_TOKEN
    )
    for special_token in special_tokens:
        word_ix[special_token] = len(word_ix)
    return word_ix

def select_batch(dataset: list, batch_ix: int, is_input: bool) -> torch.Tensor:
    """
    Select the indexed batch from the dataset, which is assumed to be padded
    :param dataset: a full dataset
    :param batch_ix: the batch index we want to select
    :param is_input: whether we want to process these sequences as inputs (as opposed to targets)
    :return: the tensor with the adjusted sequences
    """
    # Select the correct batch
    selected_batch = dataset[batch_ix]

    # Convert to inputs or targets
    truncated = [seq[:len(seq)-1] if is_input else seq[1:] for seq in selected_batch]

    # Convert to a PyTorch tensor
    return torch.tensor(truncated)

def get_n_datapoints(dataset: torch.Tensor) -> int:
    """
    Get the number of datapoints in a dataset
    :param dataset: a dataset of sequences
    :return: the number of datapoints in the dataset
    """
    return len(dataset)

def train_batch(
        model: nn.Module,
        dataset: list,
        batch_ix: int,
        optimizer: nn.Module,
        clip_gradients: bool,
        original_sequence_lengths: list
) -> float:
    """
    Train a model on a single batch
    :param model: the model to be trained
    :param dataset: the dataset in tensor format with tokens converted to indices
    :param batch_ix: the index of the batch we want to use for training
    :param optimizer: the optimizer used for training
    :param clip_gradients: whether we want to clip the gradients
    :param original_sequence_lengths: a list of lists of sequence lengths
    :return: the average sample loss for this batch
    """
    # Pytorch accumulates gradients. We need to clear them out before each training instance
    model.train()
    model.zero_grad()

    # Select the right batch and remove the first or last token for inputs or outputs
    X = select_batch(dataset, batch_ix, is_input=True)
    Y = select_batch(dataset, batch_ix, is_input=False)
    original_input_sequence_lengths = torch.tensor([seq_len - 1 for seq_len in original_sequence_lengths[batch_ix]])

    # Run our forward pass. The output is a tensor because we are using batching
    partial_pred_scores = model(X, original_input_sequence_lengths)

    # Compute the loss, gradients
    loss = model.loss(Y, partial_pred_scores.permute(0, 2, 1))
    loss.backward()

    # Clip gradients to avoid explosions
    if clip_gradients:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    # update the parameters
    optimizer.step()

    return loss.item()

def validate_batch(
        model: nn.Module,
        dataset: list,
        batch_ix: int,
        original_sequence_lengths: list
) -> float:
    # Put the model in evaluation mode
    model.eval()

    # Select the right batch and remove the first or last token for inputs or outputs
    X = select_batch(dataset, batch_ix, is_input=True)
    Y = select_batch(dataset, batch_ix, is_input=False)
    original_input_sequence_lengths = torch.tensor([seq_len - 1 for seq_len in original_sequence_lengths[batch_ix]])

    # Run our forward pass
    partial_pred_scores = model(X, original_input_sequence_lengths)

    # Compute the loss
    loss = model.loss(Y, partial_pred_scores.permute(0, 2, 1))

    return loss.item()

def pad_batch(sequences: list) -> list:
    """
    Given a list of sequences, pad all but one to have the same length as the longest one
    :param sequences: a list of sequences
    :return: the same sequences with a padding symbol added accordingly
    """
    max_length = max([len(el) for el in sequences])
    padded = [seq + [data.PAD_TOKEN] * (max_length - len(seq)) for seq in sequences]
    return padded

def batch(dataset: list, batch_size: int, word_index: dict) -> tuple:
    """
    Breaks up a dataset into batches and puts them in tensor format for PyTorch to train
    :param dataset: a list of sequences, each of which is a list of tokens
    :param batch_size: the desired batch size
    :param word_index: a map from words to indices
    :return: a tensor containing the entire dataset separated into batches, with appropriate padding
    """
    # Break up into batches
    batches = [dataset[batch_size*i:batch_size*(i+1)] for i in range(int(np.ceil(len(dataset)/batch_size)))]

    # Sort sequences in each batch from longest to shortest
    sorted_batches = [sorted(b, key=lambda seq: -len(seq)) for b in batches]

    # Add start- and end-of-sentence tokens
    enclosed = [[[data.START_OF_VERSE_TOKEN] + seq + [data.END_OF_VERSE_TOKEN] for seq in b] for b in sorted_batches]
    original_sequence_lengths = [[len(seq) for seq in b] for b in enclosed]

    # Pad inside each batch using a padding token
    padded_batches = [pad_batch(b) for b in enclosed]

    # Convert words to indices
    as_indices = [[[word_index[w] if w in word_index else word_index[data.UNKNOWN_TOKEN] for w in seq] for seq in b] \
                  for b in padded_batches]

    return as_indices, original_sequence_lengths
    # TODO: this function might belong to the data module

def get_n_batches(dataset: list) -> int:
    """
    From the relevant dimension, extract the number of batches
    :param dataset: a dataset as returned by the batch method, in tensor format
    :return: the number of batches
    """
    # TODO: this function might belong to the data module
    return len(dataset)

def train_(model: nn.Module,
           corpus: list,
           word_ix: dict,
           optimizer,
           verbose: bool,
           validate: bool,
           validation_set: list,
           config: TrainConfig
           ) -> tuple:
    epoch_train_loss, epoch_val_loss = [], []
    n_epochs = config.n_epochs

    X_train_batched, original_sequence_lengths = batch(corpus, config.batch_size, word_ix)
    n_batches_train = get_n_batches(X_train_batched)

    for epoch in range(n_epochs):
        if verbose and (int(n_epochs/10) == 0 or epoch % int(n_epochs/10) == 0):
            print(f'INFO: processing epoch {epoch}')
        batch_losses = []
        for batch_ix in range(n_batches_train):
            batch_losses.append(
                train_batch(
                    model,
                    X_train_batched,
                    batch_ix,
                    optimizer,
                    config.clip_gradients,
                    original_sequence_lengths
                )
            )

        if validate:
            epoch_val_loss.append(validate_(model, validation_set, word_ix, config.batch_size))

        # TODO: consider computing the absolute batch loss, and not the average verse loss, then divide by corpus size
        avg_sentence_loss = sum(batch_losses) / n_batches_train
        epoch_train_loss.append(avg_sentence_loss)

    # Set the size of the training set in the model and the number of epochs
    model.train_size = len(corpus)
    model.n_epochs = n_epochs

    return epoch_train_loss, epoch_val_loss

def validate_(model: nn.Module, validation_set: list, word_ix: dict, batch_size: int) -> float:

    X_val_batched, original_sequence_lengths = batch(validation_set, batch_size, word_ix)
    n_batches_val = get_n_batches(X_val_batched)

    loss = 0
    with torch.no_grad():
        for batch_ix in range(n_batches_val):
            loss += validate_batch(model, X_val_batched, batch_ix, original_sequence_lengths)

    # Set the size of the validation set in the model
    model.validation_size = len(validation_set)

    # Compute the average loss per sequence
    # TODO: consider computing the absolute batch loss, and not the average verse loss, then divide by corpus size
    avg_sentence_loss = loss / n_batches_val

    return avg_sentence_loss

def pred_sample(model: nn.Module, sample: list, word_ix: dict, ix_word: dict) -> np.ndarray:
    # Put the model in evaluation mode
    model.eval()

    words = sample.copy()
    for i in range(1, len(sample)):
        # Batching is obligatory with my model
        seq = torch.tensor([prepare_sequence(words, word_ix)], dtype=torch.long)
        original_input_sequence_lengths = torch.tensor([len(seq[0])])
        trained_next_word_scores = model(seq, original_input_sequence_lengths)[0]

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

def initialize_model(word_index: dict, config: TrainConfig) -> tuple:
    loss_function = nn.CrossEntropyLoss(ignore_index=word_index[data.PAD_TOKEN])
    model = LSTMLanguageModel(config.embedding_dim, config.hidden_dim, word_index, config.n_layers, loss_function)
    lr = config.learning_rate
    optimizer_name = config.optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_name}')
    return model, optimizer

def plot_losses(dataset_epoch_losses: dict) -> None:
    assert len(set([len(losses) for losses in dataset_epoch_losses.values()])) == 1
    for dataset, losses in dataset_epoch_losses.items():
        plt.plot(range(len(losses)), losses, label=dataset)
    plt.xlabel('Epoch')
    plt.ylabel('Avg verse loss')
    plt.legend()
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
    dataset_epoch_losses = {}
    for i in range(int(len(lines) / 2)):
        dataset_epoch_losses[lines[2*i].strip()] = [float(el.strip()) for el in lines[2*i+1].split(',')]
    return dataset_epoch_losses

def get_perplexity(loss: float) -> float:
    """
    Computes the perplexity given the loss. It is equivalent to torch.exp(loss_tensor).item()
    :param loss: the loss computed from a language model
    :return: the perplexity of the language model
    """
    return np.exp(loss)


def to_train_config(config: configparser.ConfigParser, version: str) -> TrainConfig:
    params = config[version]
    return TrainConfig(
        int(params['embedding_dim']),
        int(params['hidden_dim']),
        int(params['n_layers']),
        float(params['learning_rate']),
        int(params['n_epochs']),
        params['clip_gradients'] == 'True',
        params['optimizer'],
        int(params['batch_size'])
    )

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print(
            'USAGE:',
            sys.argv[0],
            '<bible_filename> <cfg_file> <cfg_name> <model_name> <output_dir> <is_debug>'
        )
        exit(-1)
    bible_filename = sys.argv[1]
    cfg_file = sys.argv[2]
    cfg_name = sys.argv[3]
    model_name = sys.argv[4]
    output_dir = sys.argv[5]
    # In debug mode, only 50 verses are used for training
    is_debug = sys.argv[6] == 'True'
    """
    bible_filename = '/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/eng-x-bible-world.txt'
    cfg_file = '/home/pablo/ownCloud/WordOrderBibles/GitHub/configs/pos_tagger.cfg'
    cfg_name = 'simpler.model.first'
    model_name = 'simpler_model_first'
    output_dir = '/home/pablo/ownCloud/WordOrderBibles/GitHub/output/'
    is_debug = False
    """

    bible_corpus = 'PBC'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    training_data = split_bible.train_data
    validation_data = split_bible.hold_out_data
    if is_debug:
        training_data = training_data[:50]
        validation_data = validation_data[:10]

    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    # Read the training configuration
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    cfg = to_train_config(cfg, cfg_name)
    cfg.save(f'{output_dir}/{model_name}.cfg')

    lm, sgd = initialize_model(word_to_ix, cfg)

    train_losses, validation_losses = train_(
        lm,
        training_data,
        word_to_ix,
        optimizer=sgd,
        verbose=True,
        validate=True,
        validation_set=validation_data,
        config=cfg
    )

    lm.save(f'{output_dir}/{model_name}.pth')
    dataset_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    save_losses(dataset_losses, f'{output_dir}/{model_name}_losses.txt')
