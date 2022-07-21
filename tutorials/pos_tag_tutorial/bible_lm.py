"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
from train import to_train_config
from tutorials.pos_tag_tutorial.simple_lm import get_word_index, invert_dict, initialize_model, train_, print_pred, \
    plot_losses

if __name__ == '__main__':
    bible_corpus = 'PBC'
    bible_filename = '/home/pablo/Documents/paralleltext/bibles/corpus/eng-x-bible-world.txt'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    training_data = split_bible.train_data[:10]

    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    # Read the training configuration
    cfg = configparser.ConfigParser()
    cfg.read('../../configs/pos_tagger.cfg')
    cfg = to_train_config(cfg, 'bible.lm')

    lm, nll_loss, sgd = initialize_model(cfg.embedding_dim, cfg.hidden_dim, len(word_to_ix), lr=cfg.learning_rate)

    losses = train_(
        lm,
        training_data,
        word_to_ix,
        n_epochs=cfg.n_epochs,
        loss_function=nll_loss,
        optimizer=sgd,
        verbose=True
    )

    print('After training:')
    print_pred(lm, training_data[:3], word_to_ix, ix_to_word)
    print('Expected results:')
    print('\n'.join([' '.join(sentence) for sentence in training_data[:3]]))

    plot_losses(losses)
