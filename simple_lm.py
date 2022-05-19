"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import data
from train import EMBEDDING_DIM, HIDDEN_DIM, prepare_sequence, next_word_target
import torch.nn as nn
import torch
import torch.nn.functional as functional
import numpy as np

class LSTMLanguageModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        next_word_space = self.hidden2word(lstm_out.view(len(sentence), -1))
        next_word_scores = functional.log_softmax(next_word_space, dim=1)
        return next_word_scores

def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}

def get_next_words(scores: torch.Tensor, ix_next_word: dict) -> np.ndarray:
    pred_ixs = scores.max(dim=1).indices.numpy()
    return np.vectorize(lambda ix: ix_next_word[ix])(pred_ixs)

if __name__ == '__main__':
    training_data = [
        "The dog ate the apple",
        "Everybody read that book"
    ]
    training_data = [sent.split() for sent in training_data]
    word_to_ix = {}
    # For each words-list (sentence) in the training_data
    for sent in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
    word_to_ix[data.UNKNOWN_TOKEN] = len(word_to_ix)
    word_to_ix[data.CHUNK_END_TOKEN] = len(word_to_ix)
    print(f'Word indices: {word_to_ix}')
    ix_to_word = invert_dict(word_to_ix)

    model = LSTMLanguageModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for next-word j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0], word_to_ix)
        untrained_next_word_scores = model(inputs)
        print(f'Predictions before training: {get_next_words(untrained_next_word_scores, ix_to_word)}')

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for training_sentence in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(training_sentence, word_to_ix)
            targets = prepare_sequence(next_word_target(training_sentence), word_to_ix)

            # Step 3. Run our forward pass.
            partial_pred_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(partial_pred_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        for training_sentence in training_data:
            inputs = prepare_sequence(training_sentence, word_to_ix)
            trained_next_word_scores = model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for next-word j
            # for word i. The predicted next-word is the maximum scoring next-word.
            # Here, we can see whether the predicted sequences
            # match the training data
            print(f'Predictions after training: {get_next_words(trained_next_word_scores, ix_to_word)}')
