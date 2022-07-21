import torch

import data
from data import SplitData
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import configparser
import sys

# Following https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class TrainedModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, n_layers: int):
        super(TrainedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) # TODO: what is a good value for embedding_dim?
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers) # TODO: what is a good value for hidden_dim?
        # n_layers = 1 Following the tutorial linked above; others used 2
        self.hidden2pred = nn.Linear(hidden_dim, vocab_size)

    # TODO: propagate hidden state between sentences? See: https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        pred_space = self.hidden2pred(lstm_out.view(len(sentence), -1))
        pred_scores = F.log_softmax(pred_space, dim=1)
        return pred_scores

    def pred(self, sentence: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_word_scores = self(sentence)
        return next_word_scores


class TrainConfig:
    def __init__(self, embedding_dim: int, hidden_dim: int, n_layers: int, learning_rate: float, n_epochs: int):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs


def prepare_sequence(seq: list, to_ix: dict) -> torch.Tensor:
    index_sequence = [to_ix[w] if w in to_ix else to_ix[data.UNKNOWN_TOKEN] for w in seq]
    return torch.tensor(index_sequence, dtype=torch.long)

def next_word_target(seq: list) -> list:
    """
    For next-word prediction / language modeling, assume that the target for each word is the next word.
    At the end of each chunk, we predict the padding token.
    TODO: if we decide to work with verses, we could have the target of the last word be <END>
    TODO: another alternative is to split the input sentence with N tokens. Input: a[:N-1]. Output: a[1:]
    :param seq: a sequence of tokens
    :return: the same sequence shifted by one slot and with a pad token at the end
    """
    return seq[1:] + [data.CHUNK_END_TOKEN]

def train_model(split_data: SplitData, cfg: TrainConfig, len_seq: int) -> TrainedModel:
    word_to_ix = split_data.train_word_to_ix
    model = TrainedModel(cfg.embedding_dim, cfg.hidden_dim, len(word_to_ix), cfg.n_layers)
    loss_function = nn.NLLLoss() # TODO: replace by Eq. 68 in Hahn et al?
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate) # TODO: replace SGD (pg. 83)
    # TODO: make learning_rate variable? Determine by validation?
    for epoch in range(cfg.n_epochs):
        training_data = split_data.shuffle_chop('train', len_seq)
        print(f'Epoch {epoch} / {cfg.n_epochs}')
        for i, sentence in enumerate(training_data):
            if i % (int(len(training_data) / 5)) == 0:
                print(f'Sentence {i} / {len(training_data)}')
            # Clear accumulated gradients before each instance
            model.zero_grad()

            # Turn our inputs into tensors of word indices
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(next_word_target(sentence), word_to_ix)

            # Run our forward pass.
            next_word_scores = model(sentence_in)

            # Compute loss and gradients, update parameters by calling optimizer.step()
            loss = loss_function(next_word_scores, targets)
            loss.backward()
            optimizer.step()
    return model

def pred(model: TrainedModel, sequences: list, word_to_ix: dict, ix_to_word: dict) -> list:
    """
    Make predictions from a trained model on a list of sequences
    :param model: a trained LSTM
    :param sequences: a list of sequences, each of which is a list of tokens
    :param word_to_ix: a dictionary from words to indices
    :param ix_to_word: a dictionary from indices to words allowed in the output
    :return: a list of predictions, each of which is a list of tokens
    """
    sentences_in = [prepare_sequence(seq, word_to_ix) for seq in sequences]
    sentences_out = [model.pred(sentence) for sentence in sentences_in]
    maximum_ixs = [torch.max(sentence, dim=1).indices for sentence in sentences_out]
    return [[ix_to_word[ix.item()] for ix in sentence] for sentence in maximum_ixs]

def to_train_config(cfg: configparser.ConfigParser, version: str) -> TrainConfig:
    params = cfg[version]
    return TrainConfig(
        int(params['embedding_dim']),
        int(params['hidden_dim']),
        int(params['n_layers']),
        float(params['learning_rate']),
        int(params['n_epochs'])
    )

if __name__ == '__main__':
    bible_corpus = 'PBC'
    bible_filename = '/home/pablo/Documents/paralleltext/bibles/corpus/eng-x-bible-world.txt'
    config_file = 'configs/pos_tagger.cfg'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    # Read the configurations
    config = configparser.ConfigParser()
    config.read(config_file)
    train_cfg = to_train_config(config, 'hahn.chopping')
    sequence_length = int(config['DEFAULT']['sequence_length']) # T from Hahn et al. TODO: make it variable

    # Train the model
    next_word_predictor = train_model(split_bible, train_cfg, sequence_length)

    # Make predictions on the holdout set
    data_holdout = split_bible.shuffle_chop('holdout', sequence_length)
    pred_holdout = pred(next_word_predictor, data_holdout, split_bible.train_word_to_ix, split_bible.train_ix_to_word)

    print('Some predictions:')
    for ix in range(5):
        print('\tReal:', ' '.join(data_holdout[ix]))
        print('\tPred:', ' '.join(pred_holdout[ix]))
        print('----------------------------------')
