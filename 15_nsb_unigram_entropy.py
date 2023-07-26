import sys
import numpy as np
import pandas as pd

import analysis
import data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'USAGE: {sys.argv[0]} <filename>')
        exit(-1)
    # Variables related to the location of the data and the type of system
    bibles_path = '/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/'
    bible_filename = sys.argv[1]
    #bible_filename = 'eng-x-bible-world.txt'
    output_path = '/home/pablo/Documents/GitHubRepos/WordOrderBibles/output/BentzEtAl/'
    # Variables related to the processing of text for GPT-2
    prompt = ''
    separator = ' '
    # Variables related to the processing of text for unigram entropies
    remove_punctuation = False
    lowercase = False

    bible = data.parse_pbc_bible(bibles_path + bible_filename)

    """For each of these hierarchical orders, we can compute the entropy per word and the unigram entropy."""
    by_bible, _, _, _, _ = bible.join_by_toc()
    by_level = {'bible': by_bible}

    eos_token = ''
    level_text = {level_name: data.join_texts_in_dict(id_texts, prompt, eos_token, separator) \
                  for level_name, id_texts in by_level.items()}

    raw_name = output_path + bible_filename
    level_entropies = {level_name: analysis.nsb_unigram_entropy(id_text,
                                                                remove_punctuation,
                                                                lowercase) \
                       for level_name, id_text in level_text.items()}

    level_avg_text_len = {level_name: np.mean([len(data.tokenize(text, remove_punctuation, lowercase)) \
                                               for text in id_text.values()]) \
                          for level_name, id_text in level_text.items()}

    # Save all these values to a Pandas dataframe that we can use to make histograms and compute statistics
    df = pd.DataFrame(columns=('level', 'n_tokens', 'H_unigram', 'id'))
    for level_name, section_entropies in level_entropies.items():
        for section_id, entropy in section_entropies.items():
            row = (level_name, len(data.tokenize(level_text[level_name][section_id],
                                                 remove_punctuation,
                                                 lowercase)),
                   entropy, str(section_id))
            df.loc[len(df)] = row

    df.to_csv(output_path + bible_filename.replace('.txt', '_entropies.csv'), index=False)
