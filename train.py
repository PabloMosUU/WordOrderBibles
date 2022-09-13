"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from data import batch, get_n_batches
import sys


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

    def loss(self, Y_true, Y_pred):
        return self.loss_function(Y_pred, Y_true)

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


    def get_perplexity(self, sequence: list) -> float:
        """
        Computes the perplexity of a sequence
        :param sequence: the sequence of (str) words for which we want to calculate the perplexity
        :return: the perplexity of the language model for this test sequence
        """
        index_seq = [self.word_index[word] for word in sequence]
        scores = self.forward(torch.tensor([index_seq]), torch.tensor([len(index_seq)]))[0]
        # normalize scores of next words
        probabilities = torch.nn.functional.log_softmax(scores, dim=1)
        # now iterate over next words and get their probabilities
        log_p_sum = 0
        for i, word in enumerate(sequence[1:]):
            log_p_sum += probabilities[i][self.word_index[word]]
        return np.exp(-log_p_sum/len(sequence)).item()


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
            avg_loss_per_token: bool
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

    def __repr__(self):
        return ', '.join([f'{k}: {v}' for k, v in self.to_dict().items()])

    def to_dict(self):
        return {'embedding_dim': self.embedding_dim, 'hidden_dim': self.hidden_dim, 'n_layers': self.n_layers,
                'learning_rate': self.learning_rate, 'n_epochs': self.n_epochs, 'clip_gradients': self.clip_gradients,
                'optimizer': self.optimizer, 'weight_decay': self.weight_decay, 'batch_size': self.batch_size,
                'dropout': self.dropout, 'verbose': self.verbose, 'gradient_logging': self.gradient_logging,
                'avg_loss_per_verse': self.avg_loss_per_token}

    def save(self, filename):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {k: str(v) for k, v in self.to_dict().items()}
        with open(filename, 'w') as f:
            config.write(f)


def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}


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

    # Log the gradients of the cost function for all model parameters
    model.log_gradients(batch_ix)

    # Clip gradients to avoid explosions
    if clip_gradients:
        clip_value = 0.1
        nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

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


def train_(model: nn.Module,
           corpus: list,
           word_ix: dict,
           optimizer,
           validate: bool,
           validation_set: list,
           config: TrainConfig
           ) -> tuple:
    epoch_train_loss, epoch_val_loss = [], []
    n_epochs = config.n_epochs

    X_train_batched, original_sequence_lengths = batch(corpus, config.batch_size, word_ix)
    n_batches_train = get_n_batches(X_train_batched)

    for epoch in range(n_epochs):
        model.epoch = epoch
        if config.verbose:
            print(f'LOG: epoch {epoch}')
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

        if config.verbose:
            print(f'LOG: train_batch_losses {batch_losses}')

        if validate:
            epoch_val_loss.append(
                _validate(model, validation_set, word_ix, config.batch_size, config.verbose, config.avg_loss_per_token)
            )

        if config.avg_loss_per_token:
            epoch_train_loss.append(sum(batch_losses) / n_batches_train)
        else:
            epoch_train_loss.append(sum(batch_losses) / len(corpus))

    # Set the size of the training set in the model and the number of epochs
    model.train_size = len(corpus)
    model.n_epochs = n_epochs

    return epoch_train_loss, epoch_val_loss

def _validate(model: nn.Module, validation_set: list, word_ix: dict, batch_size: int, verbose: bool,
              avg_loss_per_token: bool) -> float:

    X_val_batched, original_sequence_lengths = batch(validation_set, batch_size, word_ix)
    n_batches_val = get_n_batches(X_val_batched)

    batch_losses = []
    with torch.no_grad():
        for batch_ix in range(n_batches_val):
            batch_losses.append(validate_batch(model, X_val_batched, batch_ix, original_sequence_lengths))

    # Set the size of the validation set in the model
    model.validation_size = len(validation_set)

    if verbose:
        print(f'LOG: validation_batch_losses {batch_losses}')

    if avg_loss_per_token:
        return sum(batch_losses) / n_batches_val
    else:
        return sum(batch_losses) / len(validation_set)


def initialize_model(word_index: dict, config: TrainConfig) -> tuple:
    reduction = 'mean' if config.avg_loss_per_token else 'sum'
    loss_function = nn.CrossEntropyLoss(ignore_index=word_index[data.PAD_TOKEN], reduction=reduction)
    model = LSTMLanguageModel(
        config.embedding_dim,
        config.hidden_dim,
        word_index,
        config.n_layers,
        loss_function,
        config.avg_loss_per_token,
        config.dropout,
        config.gradient_logging
    )
    lr = config.learning_rate
    optimizer_name = config.optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_name}')
    return model, optimizer

def plot_losses(dataset_epoch_losses: dict, avg_loss_per_token: bool) -> None:
    assert len(set([len(losses) for losses in dataset_epoch_losses.values()])) == 1
    for dataset, losses in dataset_epoch_losses.items():
        plt.plot(range(len(losses)), losses, label=dataset)
    plt.xlabel('Epoch')
    plt.ylabel(f'Avg {"token" if avg_loss_per_token else "sentence"} loss')
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
        float(params['weight_decay']),
        int(params['batch_size']),
        float(params['dropout']),
        params['verbose'] == 'True',
        params['gradient_logging'] == 'True',
        params['avg_loss_per_token'] == 'True'
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
    cfg_name = 'simple.lm'
    model_name = 'simple_lm'
    output_dir = '/home/pablo/ownCloud/WordOrderBibles/GitHub/output/'
    is_debug = True
    """

    bible_corpus = 'PBC'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    training_data = split_bible.train_data
    validation_data = split_bible.hold_out_data
    if is_debug:
        training_data, validation_data = [[sent[:3] for sent in data_segment[:50]] \
                                          for data_segment in (training_data, validation_data)]

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
        validate=True,
        validation_set=validation_data,
        config=cfg
    )

    lm.save(f'{output_dir}/{model_name}.pth')
    dataset_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    save_losses(dataset_losses, f'{output_dir}/{model_name}_losses.txt')
