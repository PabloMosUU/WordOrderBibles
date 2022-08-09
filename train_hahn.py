import torch

import data
from data import SplitData, prepare_sequence
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# Following https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
from train import TrainConfig


class TrainedModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, n_layers: int):
        super(TrainedModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        # n_layers = 1 Following the tutorial linked above; others used 2
        self.hidden2pred = nn.Linear(hidden_dim, vocab_size)

    # TODO: propagate hidden state between sentences? See: https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        pred_space = self.hidden2pred(lstm_out.view(len(sentence), -1))
        pred_scores = func.log_softmax(pred_space, dim=1)
        return pred_scores

    def pred(self, sentence: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_word_scores = self(sentence)
        return next_word_scores


def next_word_target(seq: list) -> list:
    """
    For next-word prediction / language modeling, assume that the target for each word is the next word.
    At the end of each chunk, we predict the padding token.
    :param seq: a sequence of tokens
    :return: the same sequence shifted by one slot and with a pad token at the end
    """
    return seq[1:] + [data.CHUNK_END_TOKEN]

def train_model(split_data: SplitData, cfg: TrainConfig, len_seq: int) -> TrainedModel:
    word_to_ix = split_data.train_word_to_ix
    model = TrainedModel(cfg.embedding_dim, cfg.hidden_dim, len(word_to_ix), cfg.n_layers)
    loss_function = nn.NLLLoss() # TODO: replace by Eq. 68 in Hahn et al?
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
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
            sentence_in = torch.tensor(prepare_sequence(sentence, word_to_ix), dtype=torch.long)
            targets = torch.tensor(prepare_sequence(next_word_target(sentence), word_to_ix), dtype=torch.long)

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
    sentences_in = [torch.tensor(prepare_sequence(seq, word_to_ix), dtype=torch.long) for seq in sequences]
    sentences_out = [model.pred(sentence) for sentence in sentences_in]
    maximum_ixs = [torch.max(sentence, dim=1).indices for sentence in sentences_out]
    return [[ix_to_word[ix.item()] for ix in sentence] for sentence in maximum_ixs]

