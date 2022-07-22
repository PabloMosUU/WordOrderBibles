"""
The idea in this program is to reproduce the tutorial provided here:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
while using my tools as much as possible
"""
from train import prepare_sequence
import torch.nn as nn
import torch
import torch.nn.functional as functional
import numpy as np

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = functional.log_softmax(tag_space, dim=1)
        return tag_scores

def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}

def get_tags(scores: torch.Tensor, ix_tag: dict) -> np.ndarray:
    tag_ixs = scores.max(dim=1).indices.numpy()
    return np.vectorize(lambda ix: ix_tag[ix])(tag_ixs)

if __name__ == '__main__':
    training_data = [
        # Tags are: DET - determiner; NN - noun; V - verb
        # For example, the word "The" is a determiner
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
    print(f'Word indices: {word_to_ix}')
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
    ix_to_tag = invert_dict(tag_to_ix)

    model = LSTMTagger(6, 6, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        untrained_tag_scores = model(inputs)
        print(f'Predictions before training: {get_tags(untrained_tag_scores, ix_to_tag)}')

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for training_sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(training_sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            partial_tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(partial_tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        trained_tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(f'Predictions after training: {get_tags(trained_tag_scores, ix_to_tag)}')
        print('Expected result: DET NOUN VERB DET NOUN')
