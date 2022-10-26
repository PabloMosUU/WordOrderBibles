# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1ntq0iIAKXnVNQsIQB2aq9f1cdkNDneN9
In this notebook I will run an entire calculation of the entropy, random entropy, and direct entropy,
as well as the difference between them. I will do it at the bible, testament, book, chapter and verse levels.

The data is from the Parallel Bible Corpus (PBC), and the model used is GPT-2.
The language is English.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import analysis
import data

if __name__ == '__main__':
    # Variables related to the location of the data and the type of system
    bibles_path = '/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/'
    bible_filename = 'eng-x-bible-world.txt'
    device = 'cpu'
    # Variables related to the processing of text for GPT-2
    prompt = '\n\n '
    separator = ' '
    add_eos_token = True
    # Variables related to the probability calculation with GPT-2
    stride = 256
    n_prompt_tokens = 2
    # Variables related to the processing of text for unigram entropies
    remove_punctuation = False
    lowercase = False

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    bible = data.parse_pbc_bible(bibles_path + bible_filename)

    """For each of these hierarchical orders, we can compute the entropy per word and the unigram entropy."""
    by_bible, by_testament, by_book, by_chapter, by_verse = bible.join_by_toc()
    by_level = {'bible': by_bible, 'testament': by_testament, 'book': by_book, 'chapter': by_chapter, 'verse': by_verse}

    eos_token = ' ' + tokenizer.eos_token if add_eos_token else ''
    level_text = {level_name: data.join_texts_in_dict(id_texts, prompt, eos_token, separator) \
                  for level_name, id_texts in by_level.items()}

    # DEBUG
    for level in ('testament', 'book', 'chapter', 'verse'):
        del level_text[level]
    level_text['bible']['bible'] = level_text['bible']['bible'][:300]
    # END DEBUG

    token_log_likelihood = data.log_likelihoods(level_text['bible']['bible'],
                                                remove_punctuation=remove_punctuation,
                                                lowercase=lowercase)

    level_entropies = {level_name: analysis.full_entropy_calculation(id_text,
                                                                     gpt2,
                                                                     tokenizer,
                                                                     stride,
                                                                     device,
                                                                     n_prompt_tokens,
                                                                     token_log_likelihood,
                                                                     remove_punctuation,
                                                                     lowercase) \
                       for level_name, id_text in level_text.items()}

    level_avg_text_len = {level_name: np.mean([len(data.tokenize(text, remove_punctuation, lowercase)) \
                                               for text in id_text.values()]) \
                          for level_name, id_text in level_text}

    """
    For each of these, we can make two figures. 
    One is a histogram of H, H_s and H_s-H, and the other is a histogram of H, H_r and H_r-H
    """
    df = pd.DataFrame(columns=['n_texts', 'H_mean', 'H_std', 'H_s_mean', 'H_s_std', 'H_r_mean', 'H_r_std',
                               'D_s_mean', 'D_s_std', 'D_r_mean', 'D_r_std'],
                      index=level_entropies.values())
    for level_name, entropies in level_entropies.items():
        H = [el[0] for el in entropies.values()]
        H_s = [el[1] for el in entropies.values()]
        H_r = [el[2] for el in entropies.values()]
        D_s = [el[1] - el[0] for el in entropies.values()]
        D_r = [el[2] - el[0] for el in entropies.values()]
        plt.hist(H, label='H', color='green')
        plt.hist(H_s, label='H_s', color='blue')
        plt.hist(D_s, label='D_s', color='red')
        plt.xlabel('entropy [bits/word]')
        plt.ylabel(f'number of {level_name}s')
        plt.legend()
        plt.show()
        plt.hist(H, label='H', color='green')
        plt.hist(H_r, label='H_r', color='blue')
        plt.hist(D_r, label='D_r', color='red')
        plt.xlabel('entropy [bits/word]')
        plt.ylabel(f'number of {level_name}s')
        plt.legend()
        plt.show()
        name_quantity = {'H': H, 'H_s': H_s, 'H_r': H_r, 'D_s': D_s, 'D_r': D_r}
        name_stat = {}
        for name, quantity in name_quantity.items():
            name_stat[name + '_mean'] = np.mean(quantity)
            name_stat[name + '_std'] = np.std(quantity)
        name_stat['n_texts'] = len(entropies.values())
        print(f'{level_name}:', ', '.join(
            [f'{name}={np.mean(quantity):.2f}+/-{np.std(quantity):.2f}' for name, quantity in name_quantity.items()]))
        df.loc[level_name] = pd.Series(name_stat)
    df['avg_text_len'] = df.index.map(level_avg_text_len)
    df.to_csv(bibles_path + bible_filename.replace('.txt', '_entropies.csv'))
