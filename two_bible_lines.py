import configparser

import torch

from train import get_word_index, invert_dict, initialize_model, save_losses, plot_losses, train_, to_train_config
from generate import print_pred
import data

if __name__ == '__main__':
    training_data = [
        'that spoken word you yourselves know which was proclaimed throughout all judea beginning from galilee '
        'after the baptism which john preached',
        'many women were there watching from afar who had followed jesus from galilee serving him'
    ]
    validation_data = [
        'the dog ate the apple',
        'everybody read that book'
    ]
    training_data = [sent.split() for sent in training_data]
    validation_data = [sent.split() for sent in validation_data]
    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    # Read configuration
    cfg = configparser.ConfigParser()
    cfg.read('configs/pos_tagger.cfg')
    cfg = to_train_config(cfg, 'simple.lm')

    lm, lm_optimizer = initialize_model(word_to_ix, cfg)

    train_losses, validation_losses = train_(
        lm,
        training_data,
        word_to_ix,
        lm_optimizer,
        validate=True,
        validation_set=validation_data,
        config=cfg
    )

    model_name = 'simple_lm'
    lm.save(f'output/{model_name}.pth')

    print('After training:')
    print_pred(lm, training_data, word_to_ix, ix_to_word)
    print('Expected results:')
    print('\n'.join([' '.join(sentence) for sentence in training_data]))

    simple_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    save_losses(simple_losses, f'output/{model_name}_losses.txt')

    if validation_losses:
        plot_losses({'train': train_losses, 'validation': validation_losses}, True)
    else:
        plot_losses({'train': train_losses}, True)

    test_seq = f'{data.START_OF_VERSE_TOKEN} this is a sentence {data.END_OF_VERSE_TOKEN}'.split()
    perplexity = lm.get_perplexity(torch.tensor(data.to_indices(test_seq, word_to_ix)))
    print(perplexity)
