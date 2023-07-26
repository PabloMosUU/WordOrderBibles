# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1ntq0iIAKXnVNQsIQB2aq9f1cdkNDneN9
In this notebook I will run an entire calculation of the entropy, random entropy, and direct entropy,
as well as the difference between them. I will do it at the bible, testament, book, chapter and verse levels.

The data is from the Parallel Bible Corpus (PBC), and the model used is GPT-2.
The language is English.
"""

import sys
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import analysis
import data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'USAGE: {sys.argv[0]} <device> <filename>')
        exit(-1)
    # Variables related to the location of the data and the type of system
    bibles_path = '/hpc/uu_ics_ads/pmosteiro/EnglishBibles/'
    device = sys.argv[1]
    bible_filename = sys.argv[2]
    output_path = '/hpc/uu_ics_ads/pmosteiro/WordOrderBibles/output/gpt2/'
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
                          for level_name, id_text in level_text.items()}

    # Save all these values to a Pandas dataframe that we can use to make histograms and compute statistics
    df = pd.DataFrame(columns=('level', 'n_tokens', 'H', 'H_s', 'H_r'))
    for level_name, section_entropies in level_entropies.items():
        for section_id, entropies in section_entropies.items():
            row = (level_name, len(data.tokenize(level_text[level_name][section_id], remove_punctuation, lowercase)),
                   entropies[0], entropies[1], entropies[2])
            df.loc[len(df)] = row

    # Compute the word-order entropies
    df['D_r'] = df['H_r'] - df['H']
    df['D_s'] = df['H_s'] - df['H']

    # Average over all texts at each level
    col_aggs = {col: ['mean', 'std'] for col in df.columns if col != 'level'}
    # Add one more column that is the number of texts for that level
    col_aggs[list(col_aggs.keys())[0]].append('count')
    stats = df.groupby('level').agg(col_aggs)

    df.to_csv(output_path + bible_filename.replace('.txt', '_entropies.csv'), index=False)
    stats.to_csv(output_path + bible_filename.replace('.txt', '_stats.csv'))
