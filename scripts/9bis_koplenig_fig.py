"""Reproduces (part of) Figure 1 in Koplenig et al. (2017).

Usage: python 9bis_koplenig_fig.py [BIBLE_DIRECTORY] [OUTPUT_FILENAME] [MISMATCHER_FILE] [OUTPUT_DIRECTORY]
Dependencies: json
Author: Pablo Mosteiro
Status: Final
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from wordorderbibles.compression_entropy import read_selected_verses, get_entropies, join_verses
import json
# TODO: this should be done by creating Token objects here and in word_pasting.py, then joining both mark_word_structure
from word_pasting import mask_word_structure


def run(filename: str,
        lowercase: bool,
        remove_mismatcher_files: bool,
        chosen_books: list,
        truncate_books: bool,
        mismatcher_path: str,
        output_directory: str) -> dict:
    """
    Main program to run the entire pipeline on a single bible
    :param filename: the file containing the bible text
    :param lowercase: whether to lowercase the text before processing
    :param remove_mismatcher_files: whether mismatcher files should be deleted after processing
    :param chosen_books: the books for which you want to compute the entropy (PBC IDs)
    :param truncate_books: whether longer books should be truncated to the length of the shortest
    :param mismatcher_path: full path to the mismatcher executable
    :param output_directory: the directory where the output should be saved
    :return: a dictionary with entropy versions and entropies, keyed by book ID
    """
    selected_book_verses, char_counter = read_selected_verses(filename,
                                                              lowercase,
                                                              chosen_books,
                                                              truncate_books)
    # Create a base filename for each book
    book_base_filename = {book_id: os.path.join(output_directory, filename.split('/')[-1] + f'_{book_id}')
                          for book_id in selected_book_verses.keys()}
    return {book_id: get_entropies(verses,
                                   book_base_filename[book_id],
                                   remove_mismatcher_files,
                                   char_counter,
                                   mismatcher_path,
                                   mask_word_structure_fn=mask_word_structure,
                                   join_verses_fn=join_verses)
            for book_id, verses in selected_book_verses.items()}


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} bibles_dir output_filename mismatcher_filename output_dir'
    bibles_dir = sys.argv[1]  # The bible filename
    output_filename = sys.argv[2]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[3]  # The filename of the mismatcher jar
    output_dir = sys.argv[4]

    files_list = os.listdir(bibles_dir)

    books = [40, 41, 42, 43, 44, 66]
    entropies = {}
    for bible_filename in files_list:
        entropies[bible_filename] = run(filename=os.path.join(bibles_dir, bible_filename).strip(),
                                        lowercase=True,
                                        remove_mismatcher_files=True,
                                        chosen_books=books,
                                        truncate_books=True,
                                        mismatcher_path=mismatcher_file,
                                        output_directory=output_dir)

    with open(output_filename, 'w') as fp:
        json.dump(entropies, fp)
