import sys
import numpy as np
import pandas as pd

import data
import compression_entropy as ce
import analysis

def full_entropy_calculation(id_text: dict,
                             remove_punct: bool,
                             lc: bool,
                             base_name: str) -> dict:
    text_id_entropies = {}
    for text_id, text in id_text.items():
        # Tokenize for the unigram entropy computations
        tokens = data.tokenize(text, remove_punct, lc)
        # Compute the entropy rate
        base_filename = f'{base_name}_{text_id}'
        H = ce.get_entropies_per_word(tokens, base_filename, remove_mismatcher_files=True)
        # Compute the unigram entropy
        H_s = analysis.unigram_entropy_direct(tokens)
        token_log_likelihood = data.log_likelihoods(text,
                                                    remove_punctuation=remove_punctuation,
                                                    lowercase=lowercase)
        H_r = analysis.unigram_entropy_by_counts(tokens, token_log_likelihood)
        text_id_entropies[text_id] = (H, H_s, H_r)
    return text_id_entropies


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'USAGE: {sys.argv[0]} <filename>')
        exit(-1)
    # Variables related to the location of the data and the type of system
    bibles_path = '/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/'
    bible_filename = sys.argv[1]
    #bible_filename = 'eng-x-bible-world.txt'
    output_path = '/home/pablo/Documents/GitHubRepos/WordOrderBibles/output/MontemurroZanette/'
    # Variables related to the processing of text for GPT-2
    prompt = ''
    separator = ' '
    # Variables related to the processing of text for unigram entropies
    remove_punctuation = False
    lowercase = False

    bible = data.parse_pbc_bible(bibles_path + bible_filename)

    """For each of these hierarchical orders, we can compute the entropy per word and the unigram entropy."""
    by_bible, _, by_book, _, _ = bible.join_by_toc()
    by_level = {'bible': by_bible, 'book': by_book}

    eos_token = ''
    level_text = {level_name: data.join_texts_in_dict(id_texts, prompt, eos_token, separator) \
                  for level_name, id_texts in by_level.items()}

    raw_name = output_path + bible_filename
    level_entropies = {level_name: full_entropy_calculation(id_text,
                                                            remove_punctuation,
                                                            lowercase,
                                                            f'{raw_name}_{level_name}') \
                       for level_name, id_text in level_text.items()}

    level_avg_text_len = {level_name: np.mean([len(data.tokenize(text, remove_punctuation, lowercase)) \
                                               for text in id_text.values()]) \
                          for level_name, id_text in level_text.items()}

    # Save all these values to a Pandas dataframe that we can use to make histograms and compute statistics
    df = pd.DataFrame(columns=('level', 'n_tokens', 'H', 'H_s', 'H_r', 'id'))
    for level_name, section_entropies in level_entropies.items():
        for section_id, entropies in section_entropies.items():
            row = (level_name, len(data.tokenize(level_text[level_name][section_id], remove_punctuation, lowercase)),
                   entropies[0], entropies[1], entropies[2], str(section_id))
            df.loc[len(df)] = row

    # Compute the word-order entropies
    df['D_r'] = df['H_r'] - df['H']
    df['D_s'] = df['H_s'] - df['H']

    df.to_csv(output_path + bible_filename.replace('.txt', '_entropies.csv'), index=False)
