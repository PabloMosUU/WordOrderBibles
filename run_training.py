"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
import train
from train import get_word_index, invert_dict, initialize_model, train_, to_train_config
import sys

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('USAGE:', sys.argv[0], '<bible_filename> <cfg_name> <model_output_filename> <losses_filename>')
        exit(-1)
    bible_filename = sys.argv[1]
    cfg_name = sys.argv[2]
    model_output_filename = sys.argv[3]
    losses_filename = sys.argv[4]

    bible_corpus = 'PBC'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    training_data = split_bible.train_data
    validation_data = split_bible.hold_out_data

    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    # Read the training configuration
    cfg = configparser.ConfigParser()
    cfg.read('configs/pos_tagger.cfg')
    cfg = to_train_config(cfg, cfg_name)

    lm, nll_loss, sgd = initialize_model(cfg.embedding_dim, cfg.hidden_dim, word_to_ix, lr=cfg.learning_rate)

    train_losses, validation_losses = train_(
        lm,
        training_data,
        word_to_ix,
        n_epochs=cfg.n_epochs,
        loss_function=nll_loss,
        optimizer=sgd,
        verbose=True,
        validate=True,
        validation_set=validation_data
    )

    lm.save(model_output_filename)
    dataset_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    train.save_losses(dataset_losses, losses_filename)
