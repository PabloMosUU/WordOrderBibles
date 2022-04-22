import sys

import torch

import data
from data import TokenizedBible, SplitData
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N_EPOCHS = 1
SEQUENCE_LENGTH = 20 # T from Hahn et al. TODO: make it variable
EMBEDDING_DIM = 300 # TODO: what is a good value?
HIDDEN_DIM = 6 # TODO: what is a good value?
N_LAYERS = 1 # Following the tutorial linked below
LEARNING_RATE = 0.1 # TODO: make variable? Determine by validation?

# Following https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class TrainedModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, n_layers: int):
        super(TrainedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.hidden2pred = nn.Linear(hidden_dim, vocab_size)

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

def prepare_sequence(seq: list, to_ix: dict) -> torch.Tensor:
    index_sequence = [to_ix[w] if w in to_ix else to_ix[data.UNKNOWN_TOKEN] for w in seq]
    return torch.tensor(index_sequence, dtype=torch.long)

def next_word_target(seq: list) -> list:
    """
    For next-word prediction, assume that the target for each word is the next word.
    At the end of each chunk, we predict the padding token.
    :param seq: a sequence of tokens
    :return: the same sequence shifted by one slot and with a pad token at the end
    """
    return seq[1:] + [data.PAD_TOKEN]

def train_model(split_data: SplitData) -> TrainedModel:
    # TODO: shuffling should be done every epoch
    training_data = split_data.shuffle_chop('train', SEQUENCE_LENGTH)
    word_to_ix = split_data.train_word_to_ix
    model = TrainedModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), N_LAYERS)
    loss_function = nn.NLLLoss() # TODO: replace by Eq. 68 in Hahn et al?
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # TODO: replace SGD (pg. 83)
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch} / {N_EPOCHS}')
        for i, sentence in enumerate(training_data):
            if i % (int(len(training_data) / 5)) == 0:
                print(f'Sentence {i} / {len(training_data)}')
                break
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

if __name__ == '__main__':
    bible_corpus = 'PBC'
    bible_filename = '/home/pablo/Documents/paralleltext/bibles/corpus/eng-x-bible-world.txt'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    # Train the model
    next_word_predictor = train_model(split_bible)

    # Make predictions on the holdout set
    data_holdout = split_bible.shuffle_chop('holdout', SEQUENCE_LENGTH)
    pred_holdout = pred(next_word_predictor, data_holdout, split_bible.train_word_to_ix, split_bible.train_ix_to_word)

    print(pred_holdout)
