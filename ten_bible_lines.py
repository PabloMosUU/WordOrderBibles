"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
import train
from train import get_word_index, initialize_model, to_train_config
from util import invert_dict

if __name__ == '__main__':
    bible_corpus = 'PBC'
    bible_filename = '/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/eng-x-bible-world.txt'

    # Read a bible and pre-process it
    pre_processed_bible = data.process_bible(bible_filename, bible_corpus)

    # Split it
    split_bible = pre_processed_bible.split(0.15, 0.1)

    training_data = split_bible.train_data[:10]
    validation_data = split_bible.hold_out_data[:10]

    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    # Read the training configuration
    cfg = configparser.ConfigParser()
    cfg.read('configs/pos_tagger.cfg')
    cfg = to_train_config(cfg, 'bible.lm')

    lm, ten_line_opt = initialize_model(word_to_ix, cfg)

    train_losses, validation_losses = train.train(
        lm,
        training_data,
        optimizer=ten_line_opt,
        validation_set=validation_data,
        config=cfg
    )

    model_name = 'ten_bible_lines'
    lm.save(f'output/{model_name}.pth')
    train.save_losses({'train': train_losses, 'validation': validation_losses},
                      f'output/{model_name}_losses.txt')
    cfg.save(f'output/{model_name}.cfg')

    print('Perplexity:')
    print('Training data:', lm.get_perplexity(training_data, False))
    print('Validation data:', lm.get_perplexity(validation_data, False))
