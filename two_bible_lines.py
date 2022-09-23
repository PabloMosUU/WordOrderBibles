import configparser

import embed
from train import get_word_index, initialize_model, save_losses, plot_losses, train, to_train_config
from util import invert_dict
from generate import print_pred

if __name__ == '__main__':
    embeddings_file = '/home/pablo/Documents/tools/Glove/glove.6B.300d.txt'

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

    # Load the pre-trained word embeddings
    pretrained_embeddings = embed.load_embeddings(embeddings_file)

    lm, lm_optimizer = initialize_model(word_to_ix, cfg)

    train_losses, validation_losses = train(
        lm,
        training_data,
        lm_optimizer,
        validation_set=validation_data,
        config=cfg,
        word_embedding=pretrained_embeddings
    )

    model_name = 'simple_lm'
    lm.save(f'output/{model_name}.pth')

    print('After training:')
    print_pred(lm, training_data, ix_to_word, pretrained_embeddings)
    print('Expected results:')
    print('\n'.join([' '.join(sentence) for sentence in training_data]))

    simple_losses = {k:v for k, v in {'train': train_losses, 'validation': validation_losses}.items() if v}
    save_losses(simple_losses, f'output/{model_name}_losses.txt')

    if validation_losses:
        plot_losses({'train': train_losses, 'validation': validation_losses}, True)
    else:
        plot_losses({'train': train_losses}, True)
