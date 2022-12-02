from collections import defaultdict

import data
import random
import os
import numpy as np
import json

def create_random_word(word: str, char_set: str) -> str:
    return ''.join([random.choice(char_set) for _ in word])


def mask_word_structure(tokenized: list, char_set: str) -> list:
    masked = []
    word_map = {}
    new_words = set([])
    for tokens in tokenized:
        masked_tokens = []
        for token in tokens:
            if token not in word_map:
                new_word = create_random_word(token, char_set)
                if new_word in new_words:
                    raise ValueError('Random word already exists')
                word_map[token] = new_word
            masked_tokens.append(word_map[token])
        masked.append(masked_tokens)
    return masked


def join_verses(verse_tokens: list, insert_spaces: bool) -> str:
    """
    Join the verses contained in a list of lists of tokens
    :param verse_tokens: the list of verses, each of which is a list of tokens; order matters
    :param insert_spaces: whether we want to insert spaces between the tokens
    :return: the concatenated string consisting of all the tokens in the original order
    """
    sep = ' ' if insert_spaces else ''
    return sep.join([sep.join(ell) for ell in verse_tokens])

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


def run_mismatcher(preprocessed_filename: str, remove_file: bool) -> list:
    mismatcher_filename = preprocessed_filename + '_mismatcher'
    os.system(f"""java -Xmx3500M -jar \
    /home/pablo/ownCloud/WordOrderBibles/Literature/ThirdRound/dataverse_files/shortestmismatcher.jar \
    {preprocessed_filename} {mismatcher_filename}""")
    with open(mismatcher_filename, 'r') as f:
        lines = f.readlines()
    if remove_file:
        os.remove(mismatcher_filename)
    return parse_mismatcher_lines(lines)


def parse_mismatcher_lines(lines: list) -> list:
    return [int(line.split('\t')[-1].strip()) for line in lines if line != '\n']

def get_entropy(mismatches: list) -> float:
    return 1 / (sum([el/np.log2(i + 2) for i, el in enumerate(mismatches[1:])]) / len(mismatches))

def get_text_length(sequences: list) -> int:
    text = join_verses(sequences, insert_spaces=True)
    return len(text)

def truncate(sequences: list, excedent: int) -> list:
    """
    Truncate a sample by removing the number of characters in the excedent
    :param sequences: a sample represented as a list of sequences, each of which is a list of tokens
    :param excedent: the excedent that should be removed
    :return: a new list of sequences, the length of which is reduced by the excedent
    """
    orig_len = get_text_length(sequences)
    desired_length = orig_len - excedent
    output = [seq.copy() for seq in sequences]
    while get_text_length(output) > desired_length:
        if len(output[-1]) > 1:
            output[-1].pop()
        else:
            output.pop()
    return output

def select_samples(sample_sequences: dict, chosen_sample_ids: list, truncate_samples: bool) -> dict:
    if not chosen_sample_ids:
        chosen_sample_ids = list(sample_sequences.keys())
    lengths = {sample_id: get_text_length(sample_sequences[sample_id]) \
               for sample_id in chosen_sample_ids \
               if sample_id in sample_sequences}
    if len(lengths) == 0:
        return {}
    minimum_length = min(lengths.values())
    differences = {sample_id: length - minimum_length for sample_id, length in lengths.items()}
    full_samples = {sample_id: sample_sequences[sample_id] \
                    for sample_id in chosen_sample_ids \
                    if sample_id in sample_sequences}
    if truncate_samples:
        return {sample_id: truncate(sequences, differences[sample_id]) for sample_id, sequences in full_samples.items()}
    return full_samples

def get_entropies(sample_verses: list,
                  base_filename: str,
                  remove_mismatcher_files: bool,
                  char_set: str) -> dict:
    """
    Get three entropies for a given sample of verses
    :param sample_verses: the (ordered) pre-processed verses contained in the original sample
    :param base_filename: the base filename to be used for the output
    :param remove_mismatcher_files: whether to delete the mismatcher files after processing
    :param char_set: the alphabet
    :return: the entropies for the given sample (e.g., chapter)
    """
    # Randomize the order of the verses in each sample
    verse_tokens = random.sample(sample_verses, k=len(sample_verses))
    # Shuffle words within each verse
    shuffled = [random.sample(words, k=len(words)) for words in verse_tokens]
    # Mask word structure
    masked = mask_word_structure(verse_tokens, char_set)
    # Put them in a dictionary
    tokens = {'orig': verse_tokens, 'shuffled': shuffled, 'masked': masked}
    # Join all verses together
    joined = {k: join_verses(v, insert_spaces=True) for k, v in tokens.items()}
    # Save these to files to run the mismatcher
    filenames = {k: to_file(v, base_filename, k) for k, v in joined.items()}
    # Run the mismatcher
    version_mismatches = {version: run_mismatcher(preprocessed_filename, remove_mismatcher_files) \
                          for version, preprocessed_filename in filenames.items()}
    # Compute the entropy
    version_entropy = {version: get_entropy(mismatches) \
                       for version, mismatches in version_mismatches.items()}
    return version_entropy

def get_word_mismatches(verse_tokens: list,
                        base_filename: str,
                        remove_mismatcher_files: bool) -> list:
    # Replace words by characters
    characterized = replace_words(verse_tokens)
    # Join all verses together
    joined = join_verses(characterized, insert_spaces=False)
    # Save these to files to run the mismatcher
    preprocessed_filename = to_file(joined, base_filename, 'orig')
    # Run the mismatcher
    mismatches = run_mismatcher(preprocessed_filename, remove_mismatcher_files)
    return mismatches

def get_entropies_per_word(sample_verses: list,
                           base_filename: str,
                           remove_mismatcher_files: bool) -> float:
    """
    Get three entropies for a given sample of verses
    :param sample_verses: the (ordered) pre-processed verses contained in the original sample
    :param base_filename: the base filename to be used for the output
    :param remove_mismatcher_files: whether to delete the mismatcher files after processing
    :return: the entropies for the given sample (e.g., chapter)
    """
    # Compute the entropy
    return get_entropy(get_word_mismatches(sample_verses, base_filename, remove_mismatcher_files))

def read_selected_verses(filename: str,
                         lowercase: bool,
                         chosen_books: list,
                         truncate_books: bool) -> tuple:
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Tokenize by splitting on spaces
    tokenized = bible.tokenize(remove_punctuation=False, lowercase=lowercase)
    # Obtain the repertoire of symbols
    char_set = ''.join(set(''.join([el for lis in tokenized.verse_tokens.values() for el in lis])))
    # Split by book
    _, _, book_verses, _, _ = data.join_by_toc(tokenized.verse_tokens)
    # Select the books we are interested in
    selected_book_verses = select_samples(book_verses, chosen_books, truncate_books)
    return selected_book_verses, char_set

def join_words(verse: list, locations: list) -> list:
    assert all([locations[i] > locations[i+1] for i in range(len(locations)-1)])
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
            bigram_positions[word + ' ' + verse[i+1]].append((j, i))
    # Now the bigram with the longest list of positions is the most frequent bigram
    top_bigram = ''
    n_pos = 0
    for bigram, positions in bigram_positions.items():
        if len(positions) > n_pos:
            top_bigram = bigram
            n_pos = len(positions)
    #print(top_bigram, n_pos)
    return merge_positions(verses, bigram_positions[top_bigram])

def create_word_pasted_sets(id_verses: dict, n_iter: int) -> dict:
    book_id_versions = {}
    for book_id, tokens in id_verses.items():
        joined_verses = [tokens]
        for n_joins in range(n_iter):
            joined_verses.append(replace_top_bigram(joined_verses[-1]))
        book_id_versions[book_id] = joined_verses
    return book_id_versions

def run_word_pasting(filename: str,
                     lowercase: bool,
                     remove_mismatcher_files: bool,
                     chosen_books: list,
                     truncate_books: bool,
                     n_iter: int,
                     output_file_path: str) -> dict:
    selected_book_verses, char_set = read_selected_verses(filename,
                                                          lowercase,
                                                          chosen_books,
                                                          truncate_books)
    book_id_versions = create_word_pasted_sets(selected_book_verses, n_iter)
    book_id_entropies = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in enumerate(n_pairs_verses):
            print(n_pairs, end='')
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_v{n_pairs}'
            n_pairs_entropies[n_pairs] = get_entropies(verse_tokens,
                                                       base_filename,
                                                       remove_mismatcher_files,
                                                       char_set)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies

def run(filename: str,
        lowercase: bool,
        remove_mismatcher_files: bool,
        chosen_books: list,
        truncate_books: bool) -> dict:
    """
    Main program to run the entire pipeline on a single bible
    :param filename: the file containing the bible text
    :param lowercase: whether to lowercase the text before processing
    :param remove_mismatcher_files: whether mismatcher files should be deleted after processing
    :param chosen_books: the books for which you want to compute the entropy (PBC IDs)
    :param truncate_books: whether longer books should be truncated to the length of the shortest
    :return: a dictionary with entropy versions and entropies, keyed by book ID
    """
    selected_book_verses, char_set = read_selected_verses(filename,
                                                          lowercase,
                                                          chosen_books,
                                                          truncate_books)
    # Create a base filename for each book
    book_base_filename = {book_id: 'output/KoplenigEtAl/' + filename.split('/')[-1] + f'_{book_id}' \
                          for book_id in selected_book_verses.keys()}
    return {book_id: get_entropies(verses,
                                    book_base_filename[book_id],
                                    remove_mismatcher_files,
                                    char_set) \
            for book_id, verses in selected_book_verses.items()}

if __name__ == '__main__':
    with open('files_list.txt', 'r') as fi:
        files = fi.readlines()
    files_with_path = ['/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/' + file.strip() for file in files]
    entropies = {}
    for ix, file_with_path in enumerate(files_with_path):
        try:
            entropies[files[ix]] = run(file_with_path,
                                       lowercase=True,
                                       remove_mismatcher_files=True,
                                       chosen_books=[40, 41, 42, 43, 44, 66],
                                       truncate_books=True)
        except Exception as e:
            print(f'ERROR: {files[ix]}')
            print(e)
            print('--------------------------')
    output_filename = f'output/KoplenigEtAl/entropies.json'
    with open(output_filename, 'w') as fp:
        json.dump(entropies, fp)
