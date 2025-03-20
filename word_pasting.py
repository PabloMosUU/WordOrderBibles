from collections import defaultdict

import json
import sys

import compression_entropy as ce
from compression_entropy import join_verses


def mask_word_structure(tokenized: list, char_str: str, char_weights: list) -> list:
    masked = []
    word_map = {}
    new_words = set([])
    for tokens in tokenized:
        masked_tokens = []
        for token in tokens:
            if token not in word_map:
                new_word = ce.create_random_word(len(token), char_str, char_weights)
                if new_word in new_words:
                    raise ValueError('Random word already exists')
                word_map[token] = new_word
            masked_tokens.append(word_map[token])
        masked.append(masked_tokens)
    return masked


def replace_words(verse_tokens: list) -> list:
    """
    Replace each word by a single character
    :param verse_tokens: a list of tokens
    :return: a list of the same length, where each token is replaced by a single character
    """
    word_char = {}
    verse_chars = []
    for token in verse_tokens:
        if token not in word_char:
            word_char[token] = chr(len(word_char))
        verse_chars.append(word_char[token])
    return verse_chars


def get_entropies(sample_verses: list,
                  base_filename: str,
                  remove_mismatcher_files: bool,
                  char_counter: dict,
                  mismatcher_path: str) -> dict:
    """
    Get three entropies for a given sample of verses
    :param sample_verses: the (ordered) pre-processed verses contained in the original sample
    :param base_filename: the base filename to be used for the output
    :param remove_mismatcher_files: whether to delete the mismatcher files after processing
    :param char_counter: the alphabet with the number of times each character is seen
    :param mismatcher_path: the path to the mismatcher Java jar file
    :return: the entropies for the given sample (e.g., chapter)
    """
    return ce.get_entropies(sample_verses, base_filename, remove_mismatcher_files, char_counter, mismatcher_path,
                            mask_word_structure, join_verses)


def get_word_mismatches(verse_tokens: list,
                        base_filename: str,
                        remove_mismatcher_files: bool,
                        mismatcher_path: str) -> list:
    # Replace words by characters
    characterized = replace_words(verse_tokens)
    # Join all verses together
    joined = join_verses(characterized, insert_spaces=False)
    # Save these to files to run the mismatcher
    preprocessed_filename = ce.to_file(joined, base_filename, 'orig')
    # Run the mismatcher
    mismatches = ce.run_mismatcher(preprocessed_filename, remove_mismatcher_files, mismatcher_path)
    return mismatches


def join_words(verse: list, locations: list) -> list:
    assert all([locations[i] > locations[i + 1] for i in range(len(locations) - 1)])
    location_set = set(locations)
    assert len(location_set) == len(locations)
    joined = []
    i = 0
    while i < len(verse):
        if i in location_set:
            joined.append(verse[i] + ' ' + verse[i + 1])
            i += 2
        else:
            joined.append(verse[i])
            i += 1
    return joined


def merge_positions(verses: list, positions: list) -> list:
    verse_locations = defaultdict(list)
    for position in positions:
        verse_locations[position[0]].append(position[1])
    verse_locations = {verse: sorted(locations, reverse=True) for verse, locations in verse_locations.items()}
    for verse_ix, locations in verse_locations.items():
        verses[verse_ix] = join_words(verses[verse_ix], locations)
    return verses


def replace_top_bigram(verses: list) -> list:
    bigram_positions = defaultdict(list)
    for j, verse in enumerate(verses):
        for i, word in enumerate(verse[:-1]):
            bigram_positions[word + ' ' + verse[i + 1]].append((j, i))
    # Now the bigram with the longest list of positions is the most frequent bigram
    if not bigram_positions:
        return []
    top_bigram = max(bigram_positions, key=lambda x: len(bigram_positions[x]))
    return merge_positions(verses, bigram_positions[top_bigram])


def create_word_pasted_sets(id_verses: dict, steps_to_save: set) -> dict:
    max_merges = max(steps_to_save)
    book_id_versions = {}
    for book_id, tokens in id_verses.items():
        joined_verses = {}
        last_version = tokens.copy()
        if 0 in steps_to_save:
            joined_verses[0] = tokens.copy()
        for n_joins in range(1, max_merges + 1):
            last_version = replace_top_bigram(last_version)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = last_version.copy()
        book_id_versions[book_id] = joined_verses
    return book_id_versions


def run_word_pasting(filename: str,
                     lowercase: bool,
                     remove_mismatcher_files: bool,
                     chosen_books: list,
                     truncate_books: bool,
                     merge_steps_to_save: set,
                     output_file_path: str,
                     mismatcher_path: str) -> dict:
    selected_book_verses, char_counter = ce.read_selected_verses(filename,
                                                                 lowercase,
                                                                 chosen_books,
                                                                 truncate_books)
    book_id_versions = create_word_pasted_sets(selected_book_verses, merge_steps_to_save)
    book_id_entropies = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            print(n_pairs, end='')
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_v{n_pairs}'
            n_pairs_entropies[n_pairs] = get_entropies(verse_tokens,
                                                       base_filename,
                                                       remove_mismatcher_files,
                                                       char_counter,
                                                       mismatcher_path)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]  # The filename of the mismatcher jar

    merge_steps = set([ell for lis in [list(el) for el in (range(0, 1000, 100), range(1000, 11000, 1000))]
                       for ell in lis])

    book_entropies = {}
    for bid in [40, 41, 42, 43, 44, 66]:
        file_book_entropies = run_word_pasting(bible_filename,
                                               lowercase=True,
                                               remove_mismatcher_files=False,
                                               chosen_books=[bid],
                                               truncate_books=False,
                                               merge_steps_to_save=merge_steps,
                                               output_file_path=temp_dir,
                                               mismatcher_path=mismatcher_file)
        if bid not in file_book_entropies:
            print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
            continue
        book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
