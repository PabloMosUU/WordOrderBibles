from collections import defaultdict

import json
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.wordorderbibles import compression_entropy as ce

MAX_MASK_TRIES = 1000

def mask_word_structure(tokenized: list, char_str: str, char_weights: list) -> list:
    masked = []
    word_map = {}
    new_words = set([])
    for tokens in tokenized:
        masked_tokens = []
        for token in tokens:
            # Leave single-character tokens unchanged
            if len(token) == 1:
                masked_tokens.append(token)
                continue

            if token not in word_map:
                attempts = 1
                new_word = ce.create_random_word(len(token), char_str, char_weights)
                while new_word in new_words and attempts < MAX_MASK_TRIES:
                    new_word = ce.create_random_word(len(token), char_str, char_weights)
                    attempts += 1

                if new_word in new_words:
                    raise ValueError(f'Could not create a unique replacement for {token} after {MAX_MASK_TRIES} attempts')
                new_words.add(new_word)
                word_map[token] = new_word
            masked_tokens.append(word_map[token])
        masked.append(masked_tokens)
    return masked


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
                            mask_word_structure, ce.join_verses)


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

#Returns top bigram as well so that it may be recorded
def replace_and_get_top_bigram(verses: list, given_bigram=None) -> list:
    bigram_positions = defaultdict(list)
    for j, verse in enumerate(verses):
        for i, word in enumerate(verse[:-1]):
            bigram_positions[word + ' ' + verse[i + 1]].append((j, i))
    # Now the bigram with the longest list of positions is the most frequent bigram
    if not bigram_positions:
        return [], given_bigram
    if given_bigram != None:
        top_bigram = given_bigram
    else:
        top_bigram = max(bigram_positions, key=lambda x: len(bigram_positions[x]))

    #If the bigram position is empty then that means this bigram doesn't appear in the test corpus and we essentially skip this merging step by just returning the input verses
    if bigram_positions[top_bigram] == []:
        return verses, top_bigram
    else:
        return merge_positions(verses, bigram_positions[top_bigram]), top_bigram

#Largely the same as create_word_pasted sets; however, meant to be run on the training set so the order of merges can be recorded
def get_merge_order(id_verses: dict, steps_to_save: set) -> dict:
    max_merges = max(steps_to_save)
    book_id_merges = {}
    for book_id, tokens in id_verses.items():
        merges = [] #top bigram at every iteration
        joined_verses = {}
        last_version = tokens.copy()
        if 0 in steps_to_save:
            joined_verses[0] = tokens.copy()
        for n_joins in range(1, max_merges + 1):
            last_version, top_bigram = replace_and_get_top_bigram(last_version)
            merges.append(top_bigram)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = last_version.copy()
        book_id_merges[book_id] = merges
    return book_id_merges


def create_pasted_test_set(id_verses: dict, steps_to_save: set, id_merges: dict) -> dict:
    max_merges = max(steps_to_save)
    book_id_versions = {}
    for book_id, tokens in id_verses.items():
        if book_id in id_verses.keys() and book_id not in id_merges.keys():
            print("Skipping one pasted test set loop!")
            continue
        joined_verses = {}
        merges = id_merges[book_id]
        last_version = tokens.copy()
        if 0 in steps_to_save:
            joined_verses[0] = tokens.copy()
        for n_joins in range(1, max_merges + 1):
            if n_joins - 1 >= len(merges): #Break in case train set is smaller than test set
                break
            bigram = merges[n_joins - 1]
            last_version, top_bigram = replace_and_get_top_bigram(last_version, bigram)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = last_version.copy()
        book_id_versions[book_id] = joined_verses
    return book_id_versions


def create_paste_files(verses):
    with open("engbasic_full_bible_pasted.txt", "w", encoding="utf-8") as f:
        for sentence in verses:
            sentence_merged = " ".join(item.replace(" ", "") for item in sentence)
            sentence_capitalized = sentence_merged[0].upper() + sentence_merged[1:] if sentence_merged else ""
            f.write(sentence_capitalized + "\n")


def run_word_pasting(filename: str,
                     lowercase: bool,
                     remove_mismatcher_files: bool,
                     chosen_books: list,
                     truncate_books: bool,
                     merge_steps_to_save: set,
                     output_file_path: str,
                     mismatcher_path: str,
                     training_books: str,
                     mode = "CROSSVAL",
                     random_seed = 42) -> dict:
    selected_book_verses, char_counter = ce.read_selected_verses(filename,
                                                                 lowercase,
                                                                 chosen_books,
                                                                 truncate_books)
    training_book_verses, training_char_counter = ce.read_selected_verses(filename,
                                                                              lowercase,
                                                                              training_books,
                                                                              truncate_books)
    train_verses = {}
    test_verses = {}
    random.seed(random_seed)
    i = 0
    for bid in chosen_books:
        verses = selected_book_verses[bid]
        n = len(verses)
        match mode:
            case "SPLIT":
                mid = n // 2
                train_verses[bid] = verses[:mid]
                test_verses[bid] = verses[mid:]
            case "RANDOM":
                train_list = set(random.sample(range(n), n // 2))
                test_list = set(range(n)) - train_list
                train_verses[bid] = [verses[i] for i in train_list]
                test_verses[bid] = [verses[i] for i in test_list]
            case "CROSSVAL":
                test_verses[bid] = verses
                print(training_books)
                all_book_verses = []
                for book in training_books:
                    all_book_verses = all_book_verses + training_book_verses[book]
                
                    train_verses[bid] = all_book_verses
        i += 1

    #TODO: fix this, currently it throws an integer is not iterable error.
    #if chosen_books == ["full_bible"]:
      #  return run_word_pasting_all(filename, lowercase, remove_mismatcher_files, truncate_books, merge_steps_to_save, output_file_path, mismatcher_path)

    chosen_merges = get_merge_order(train_verses, merge_steps_to_save)
    train_book_id_versions = create_pasted_test_set(test_verses, merge_steps_to_save, chosen_merges)
    book_id_entropies = {}

    for book_id, n_pairs_verses in train_book_id_versions.items():
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            base_filename = os.path.join(temp_dir, os.path.basename(bible_filename))
            n_pairs_entropies[n_pairs] = get_entropies(verse_tokens,
                                                       base_filename,
                                                       remove_mismatcher_files,
                                                       char_counter,
                                                       mismatcher_path)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies

# TODO: is it not possible to do this with the existing methods, passing the list of books?
# TODO: update this code to fully work with the partitioning modes. However, as stated above, this may not be necessary.
def run_word_pasting_all(filename: str,
                         lowercase: bool,
                         remove_mismatcher_files: bool,
                         truncate_books: bool,
                         merge_steps_to_save: set,
                         output_file_path: str,
                         mismatcher_path: str,
                         ) -> dict:

    train_verses = {}
    test_verses = {}
    all_verses_dict, char_counter = ce.read_all_verses(filename, lowercase, truncate_books)
    verses = all_verses_dict['full_bible']
    mid = len(verses) // 2
    train_verses['full_bible'] = verses[:mid]
    test_verses['full_bible'] = verses[mid:]
    chosen_merges = get_merge_order(train_verses, merge_steps_to_save)
    n_pairs_versions = create_pasted_test_set(test_verses, merge_steps_to_save, chosen_merges)

    n_pairs_entropies = {}
    for n_pairs, verse_tokens in n_pairs_versions.items():
        create_paste_files(verse_tokens)
        base_filename = f'{output_file_path}/{filename.split("/")[-1]}_fullbibles_v{n_pairs}'
        n_pairs_entropies[n_pairs] = get_entropies(
            verse_tokens,
            base_filename,
            remove_mismatcher_files,
            char_counter,
            mismatcher_path
        )

    return {'full_bible': n_pairs_entropies}


if __name__ == '__main__':
    assert len(sys.argv) == 6, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]  # The filename of the mismatcher jar
    mode = str(sys.argv[5]) # mode in which to run the program, either crossval, split or random.

    merge_steps = set([ell for lis in [list(el) for el in (range(0, 1000, 100), range(1000, 11000, 1000))]
                       for ell in lis])
    
    book_entropies = {}
    bids = [40, 41, 42, 43, 44, 66]
    mode = mode.upper()
    if mode not in ["SPLIT", "RANDOM", "CROSSVAL"]:
        print(mode)
        raise ValueError("Please enter a valid mode: split, random or crossval.")
    book_entropies = {}
        
    if mode == "CROSSVAL":
            j = 0
            for j, bid in enumerate(bids):
                    test_book = bids[j]
                    training_books =  bids.copy();
                    del training_books[j]  
                    file_book_entropies = run_word_pasting(bible_filename,
                                                             lowercase=True,
                                                             remove_mismatcher_files=False,
                                                             chosen_books=[test_book],
                                                             truncate_books=False,
                                                             merge_steps_to_save=merge_steps,
                                                             output_file_path=temp_dir,
                                                             mismatcher_path=mismatcher_file,
                                                             mode="CROSSVAL",
                                                             training_books=training_books)
                    if bid not in file_book_entropies:
                        print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
                        continue
                    book_entropies[bid] = file_book_entropies[bid]
    else:
        for bid in [40, 41, 42, 43, 44, 66]:
            file_book_entropies = run_word_pasting(bible_filename,
                                                lowercase=True,
                                                remove_mismatcher_files=False,
                                                chosen_books=[bid],
                                                truncate_books=False,
                                                merge_steps_to_save=merge_steps,
                                                output_file_path=temp_dir,
                                                mismatcher_path=mismatcher_file,
                                                mode="mode",
                                                training_books=None)
            if bid not in file_book_entropies:
                print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
                continue
            book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
