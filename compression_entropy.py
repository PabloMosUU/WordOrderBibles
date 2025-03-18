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


def get_entropies(sample_verses: list,
                  base_filename: str,
                  remove_mismatcher_files: bool,
                  char_counter: dict,
                  mismatcher_path: str,
                  mask_word_structure_fn,
                  join_verses_fn) -> dict:
    """
    Get three entropies for a given sample of verses
    :param sample_verses: the (ordered) pre-processed verses contained in the original sample
    :param base_filename: the base filename to be used for the output
    :param remove_mismatcher_files: whether to delete the mismatcher files after processing
    :param char_counter: the alphabet with the number of times each character is seen
    :param mismatcher_path: the path to the mismatcher Java jar file
    :param mask_word_structure_fn: the function that should be used for masking word structure
    :param join_verses_fn: the function that should be used for joining verses
    :return: the entropies for the given sample (e.g., chapter)
    """
    # Randomize the order of the verses in each sample
    verse_tokens = random.sample(sample_verses, k=len(sample_verses))
    # Shuffle words within each verse
    shuffled = [random.sample(words, k=len(words)) for words in verse_tokens]
    # Mask word structure
    char_str = ''.join(char_counter.keys())
    char_weights = [char_counter[el] for el in char_str]
    masked = mask_word_structure_fn(verse_tokens, char_str, char_weights)
    # Put them in a dictionary
    tokens = {'orig': verse_tokens, 'shuffled': shuffled, 'masked': masked}
    # Join all verses together
    joined = {k: join_verses_fn(v, insert_spaces=True) for k, v in tokens.items()}
    # Save these to files to run the mismatcher
    filenames = {k: to_file(v, base_filename, k) for k, v in joined.items()}
    # Run the mismatcher
    version_mismatches = {version: run_mismatcher(preprocessed_filename,
                                                  remove_mismatcher_files,
                                                  mismatcher_path)
                          for version, preprocessed_filename in filenames.items()}
    # Compute the entropy
    version_entropy = {version: get_entropy(mismatches)
                       for version, mismatches in version_mismatches.items()}
    return version_entropy
