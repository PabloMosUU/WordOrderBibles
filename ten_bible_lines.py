"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import configparser

import data
from LstmLM import train
from LstmLM.train import get_word_index, invert_dict, initialize_model, train_, to_train_config

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

    train_losses, validation_losses = train_(
        lm,
        training_data,
        word_to_ix,
        optimizer=ten_line_opt,
        validate=True,
        validation_set=validation_data,
        config=cfg
    )

    model_name = 'ten_bible_lines'
    lm.save(f'output/{model_name}.pth')
    train.save_losses({'train': train_losses, 'validation': validation_losses},
                      f'output/{model_name}_losses.txt')
    cfg.save(f'output/{model_name}.cfg')
