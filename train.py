import torch

import data
from data import TokenizedBible, SplitData
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N_EPOCHS = 300
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

def prepare_sequence(seq: list, to_ix: dict) -> torch.Tensor:
    index_sequence = [to_ix[w] for w in seq]
    return torch.tensor(index_sequence, dtype=torch.long)

def next_word_target(seq: list) -> list:
    """
    For next-word prediction, assume that the target for each word is the next word.
    At the end of each chunk, we predict the padding token.
    :param seq: a sequence of tokens
    :return: the same sequence shifted by one slot and with a pad token at the end
    """
    return seq[1:] + data.PAD_TOKEN

def train_model(split_data: SplitData) -> TrainedModel:
    # TODO: shuffling should be done every epoch
    training_data = split_data.shuffle_chop('train', SEQUENCE_LENGTH)
    word_to_ix = split_data.train_word_to_ix
    model = TrainedModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), N_LAYERS)
    loss_function = nn.NLLLoss() # TODO: replace by Eq. 68 in Hahn et al?
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # TODO: replace SGD (pg. 83)
    for epoch in range(N_EPOCHS):
        for sentence in training_data:
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

if __name__ == '__main__':

    # Now I have a framework for training an LSTM on one of my bibles
    # I'd like to read one of these bibles

    tokenized_bible = None # read from somewhere
    split_bible = tokenized_bible.split(hold_out_fraction, test_fraction)
    hold_out_fraction = 0.15
    test_fraction = 0.1
    trained_model = train_model(split_bible)

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)