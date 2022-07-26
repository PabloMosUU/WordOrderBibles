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
import sys

class LSTMLanguageModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_index: dict, n_layers: int):
        super(LSTMLanguageModel, self).__init__()
        self.word_index = word_index
        vocab_size = len(self.word_index)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers)

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        # Variables to store the number of training and validation data points and the number of epochs
        self.train_size = None
        self.validation_size = None
        self.n_epochs = None

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

def train_batch(
        model: nn.Module,
        dataset: torch.Tensor,
        batch_ix: int,
        word_ix: dict,
        loss_function: nn.Module,
        optimizer: nn.Module,
        clip_gradients: bool
) -> float:
    """
    Train a model on a single batch
    :param model: the model to be trained
    :param dataset: the dataset
    :param batch_ix: the index of the batch we want to use for training
    :param word_ix: a map from words to indices
    :param loss_function: the loss function we want to minimize
    :param optimizer: the optimizer used for training
    :param clip_gradients: whether we want to clip the gradients
    :return: the average sample loss for this batch
    """
    raise NotImplementedError()

def train_sample_(
        model: nn.Module,
        sample: list,
        word_ix: dict,
        loss_function,
        optimizer,
        clip_gradients: bool
) -> float:
    # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
    model.train()
    model.zero_grad()

    # Step 2. Get our inputs ready for the network, that is, turn them into tensors of word indices.
    sentence_in = prepare_sequence([data.START_OF_VERSE_TOKEN] + sample, word_ix)
    targets = prepare_sequence(sample + [data.END_OF_VERSE_TOKEN], word_ix)

    # Step 3. Run our forward pass.
    partial_pred_scores = model(sentence_in)

    # Step 4. Compute the loss, gradients
    loss = loss_function(partial_pred_scores, targets)
    loss.backward()

    # Clip gradients to avoid explosions
    if clip_gradients:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    # update the parameters
    optimizer.step()

    return loss.item()

def validate_sample_(model: nn.Module, sample: list, word_ix: dict, loss_function: nn.Module) -> float:
    # Put the model in evaluation mode
    model.eval()

    # Get our inputs ready for the network, that is, turn them into tensors of word indices.
    sentence_in = prepare_sequence([data.START_OF_VERSE_TOKEN] + sample, word_ix)
    targets = prepare_sequence(sample + [data.END_OF_VERSE_TOKEN], word_ix)

    # Run our forward pass.
    partial_pred_scores = model(sentence_in)

    # Compute the loss
    loss = loss_function(partial_pred_scores, targets)

    return loss.item()

def batch(dataset: list, batch_size: int) -> torch.Tensor:
    """
    Breaks up a dataset into batches and puts them in tensor format for PyTorch to train
    :param dataset: a list of sequences, each of which is a list of tokens
    :param batch_size: the desired batch size
    :return: a tensor containing the entire dataset separated into batches, with appropriate padding
    """
    # Add start- and end-of-sentence tokens? [Maybe better to do it one level above]
    # Break up into batches
    # Pad inside each batch using a padding token
    # Create a PyTorch tensor out of this dataset
    # TODO: this function might belong to the data module
    raise NotImplementedError()

def get_n_batches(dataset: torch.Tensor) -> int:
    """
    From the relevant dimension, extract the number of batches
    :param dataset: a dataset as returned by the batch method, in tensor format
    :return: the number of batches
    """
    # TODO: this function might belong to the data module
    raise NotImplementedError()

def train_(model: nn.Module,
           corpus: list,
           word_ix: dict,
           loss_function,
           optimizer,
           verbose: bool,
           validate: bool,
           validation_set: list,
           config: TrainConfig
           ) -> tuple:
    epoch_train_loss, epoch_val_loss = [], []
    n_epochs = config.n_epochs

    X_train_batched = batch(corpus, config.batch_size)
    n_batches_train = get_n_batches(X_train_batched)

    for epoch in range(n_epochs):
        if verbose and (int(n_epochs/10) == 0 or epoch % int(n_epochs/10) == 0):
            print(f'INFO: processing epoch {epoch}')
        batch_losses = []
        for batch_ix in range(n_batches_train):
            batch_losses.append(
                train_batch(model, X_train_batched, batch_ix, word_ix, loss_function, optimizer, config.clip_gradients)
            )
        if validate:
            epoch_val_loss.append(validate_(model, validation_set, word_ix, loss_function))

        # TODO: consider computing the absolute batch loss, and not the average verse loss, then divide by corpus size
        avg_sentence_loss = sum(batch_losses) / n_batches_train
        epoch_train_loss.append(avg_sentence_loss)

    # Set the size of the training set in the model and the number of epochs
    model.train_size = len(corpus)
    model.n_epochs = n_epochs

    return epoch_train_loss, epoch_val_loss

def validate_(model: nn.Module, validation_set: list, word_ix: dict, loss_function: nn.Module) -> float:
    loss = 0
    with torch.no_grad():
        for validation_sentence in validation_set:
            loss += validate_sample_(model, validation_sentence, word_ix, loss_function)

    # Set the size of the validation set in the model
    model.validation_size = len(validation_set)

    return loss / len(validation_set)

def pred_sample(model: nn.Module, sample: list, word_ix: dict, ix_word: dict) -> np.ndarray:
    # Put the model in evaluation mode
    model.eval()

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

def initialize_model(word_index: dict, config: TrainConfig) -> tuple:
    model = LSTMLanguageModel(config.embedding_dim, config.hidden_dim, word_index, config.n_layers)
    loss_function = nn.CrossEntropyLoss()
    lr = config.learning_rate
    optimizer_name = config.optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_name}')
    return model, loss_function, optimizer

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


def prepare_sequence(seq: list, to_ix: dict) -> torch.Tensor:
    index_sequence = [to_ix[w] if w in to_ix else to_ix[data.UNKNOWN_TOKEN] for w in seq]
    return torch.tensor(index_sequence, dtype=torch.long)


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

    lm, nll_loss, sgd = initialize_model(word_to_ix, cfg)

    train_losses, validation_losses = train_(
        lm,
        training_data,
        word_to_ix,
        loss_function=nll_loss,
        optimizer=sgd,
        verbose=True,
        validate=True,
        validation_set=validation_data,
        config=cfg
    )

    lm.save(f'{output_dir}/{model_name}.pth')
    dataset_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    save_losses(dataset_losses, f'{output_dir}/{model_name}_losses.txt')
