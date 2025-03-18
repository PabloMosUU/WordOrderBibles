import os
import random

import numpy as np

import data
from word_pasting import get_char_distribution, select_samples, parse_mismatcher_lines


def read_selected_verses(filename: str,
                         lowercase: bool,
                         chosen_books: list,
                         truncate_books: bool) -> tuple:
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Tokenize by splitting on spaces
    tokenized = bible.tokenize(remove_punctuation=False, lowercase=lowercase)
    # Obtain the repertoire of symbols
    char_counter = get_char_distribution(''.join([el for lis in tokenized.verse_tokens.values() for el in lis]))
    # Split by book
    _, _, book_verses, _, _ = data.join_by_toc(tokenized.verse_tokens)
    # Select the books we are interested in
    selected_book_verses = select_samples(book_verses, chosen_books, truncate_books)
    return selected_book_verses, char_counter


def run_mismatcher(preprocessed_filename: str, remove_file: bool, executable_path: str) -> list:
    mismatcher_filename = preprocessed_filename + '_mismatcher'
    os.system(f"""java -Xmx3500M -jar \
                {executable_path} {preprocessed_filename} {mismatcher_filename}""")
    with open(mismatcher_filename, 'r') as f:
        lines = f.readlines()
    if remove_file:
        os.remove(mismatcher_filename)
    return parse_mismatcher_lines(lines)


def get_entropy(mismatches: list) -> float:
    return 1 / (sum([el / np.log2(i + 2) for i, el in enumerate(mismatches[1:])]) / len(mismatches))


def to_file(text: str, base_filename: str, appendix: str) -> str:
    """
    Save a text to a file
    :param text: the text to be saved
    :param base_filename: base filename to be used
    :param appendix: appendix to be added to this filename
    :return: the new filename created
    """
    dot_parts = base_filename.split('.')
    extension = dot_parts[-1]
    prefix = '.'.join(dot_parts[:-1])
    new_filename = prefix + '_' + appendix + '.' + extension
    with open(new_filename, 'w') as f:
        f.write(text)
    return new_filename


def create_random_word(word_length: int, char_repertoire: str, weights: list) -> str:
    assert len(char_repertoire) == len(weights)
    return ''.join(random.choices(char_repertoire, weights=weights, k=word_length))
