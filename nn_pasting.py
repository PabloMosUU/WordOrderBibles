from collections import defaultdict, Counter

import spacy

import data
import random
import os
import numpy as np
import json
import sys


class TaggedWord:
    def __init__(self, word: str, pos: str):
        self.word = word
        self.pos = pos

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos

    def __repr__(self):
        return f'({self.word}, {self.pos})'


def create_random_word(word_length: int, char_repertoire: str, weights: list) -> str:
    assert len(char_repertoire) == len(weights)
    return ''.join(random.choices(char_repertoire, weights=weights, k=word_length))


def mask_word_structure(tokenized: list, char_str: str, char_weights: list) -> list:
    masked = []
    word_map = {}
    new_words = set([])
    for tokens in tokenized:
        masked_tokens = []
        for token in tokens:
            if token not in word_map:
                new_word = create_random_word(len(token), char_str, char_weights)
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


# TODO: remove files after running
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


def run_mismatcher(preprocessed_filename: str, remove_file: bool, executable_path: str) -> list:
    mismatcher_filename = preprocessed_filename + '_mismatcher'
    os.system(f"""java -Xmx3500M -jar \
                {executable_path} {preprocessed_filename} {mismatcher_filename}""")
    with open(mismatcher_filename, 'r') as f:
        lines = f.readlines()
    if remove_file:
        os.remove(mismatcher_filename)
    return parse_mismatcher_lines(lines)


def parse_mismatcher_lines(lines: list) -> list:
    return [int(line.split('\t')[-1].strip()) for line in lines if line != '\n']


def get_entropy(mismatches: list) -> float:
    return 1 / (sum([el / np.log2(i + 2) for i, el in enumerate(mismatches[1:])]) / len(mismatches))


def get_text_length(sequences: list) -> int:
    text = join_verses(sequences, insert_spaces=True)
    return len(text)


def truncate(sequences: list, surplus: int) -> list:
    """
    Truncate a sample by removing the number of characters in the surplus
    :param sequences: a sample represented as a list of sequences, each of which is a list of tokens
    :param surplus: the surplus that should be removed
    :return: a new list of sequences, the length of which is reduced by the surplus
    """
    orig_len = get_text_length(sequences)
    desired_length = orig_len - surplus
    output = [seq.copy() for seq in sequences]
    while get_text_length(output) > desired_length:
        if len(output[-1]) > 1:
            output[-1].pop()
        else:
            output.pop()
    return output


def select_samples(bible: data.PbcBible, chosen_sample_ids: list[int]) -> dict[int, list[str]]:
    # If no book IDs are passed, keep them all
    if not chosen_sample_ids:
        chosen_sample_ids = set([el[:2] for el in bible.content.keys()])
    # Parse all verses and keep only those in relevant books
    last_code = "00000000"
    by_book = defaultdict(list)
    for code, text in bible.content.items():
        if text.strip() == '':
            continue
        assert code >= last_code, f'The verses are not ordered by verse ID: ({last_code}, {code})'
        last_code = code
        book = int(code[:2])
        if book not in chosen_sample_ids:
            continue
        by_book[book].append(text)
    # Check that at least one book has verses
    lengths = {sample_id: get_text_length(verses)
               for sample_id, verses in by_book.items()}
    if len(lengths) == 0:
        return {}
    return by_book


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
    # Randomize the order of the verses in each sample
    verse_tokens = random.sample(sample_verses, k=len(sample_verses))
    # Shuffle words within each verse
    shuffled = [random.sample(words, k=len(words)) for words in verse_tokens]
    # Mask word structure
    char_str = ''.join(char_counter.keys())
    char_weights = [char_counter[el] for el in char_str]
    masked = mask_word_structure(verse_tokens, char_str, char_weights)
    # Put them in a dictionary
    tokens = {'orig': verse_tokens, 'shuffled': shuffled, 'masked': masked}
    # Join all verses together
    joined = {k: join_verses(v, insert_spaces=True) for k, v in tokens.items()}
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


def get_word_mismatches(verse_tokens: list,
                        base_filename: str,
                        remove_mismatcher_files: bool,
                        mismatcher_path: str) -> list:
    # Replace words by characters
    characterized = replace_words(verse_tokens)
    # Join all verses together
    joined = join_verses(characterized, insert_spaces=False)
    # Save these to files to run the mismatcher
    preprocessed_filename = to_file(joined, base_filename, 'orig')
    # Run the mismatcher
    mismatches = run_mismatcher(preprocessed_filename, remove_mismatcher_files, mismatcher_path)
    return mismatches


# TODO: get rid of this nearly trivial function
def get_entropies_per_word(sample_verses: list,
                           base_filename: str,
                           remove_mismatcher_files: bool,
                           mismatcher_path: str) -> float:
    """
    Get three entropies for a given sample of verses
    :param sample_verses: the (ordered) pre-processed verses contained in the original sample
    :param base_filename: the base filename to be used for the output
    :param remove_mismatcher_files: whether to delete the mismatcher files after processing
    :param mismatcher_path: path to mismatcher jar
    :return: the entropies for the given sample (e.g., chapter)
    """
    # Compute the entropy
    return get_entropy(get_word_mismatches(sample_verses, base_filename, remove_mismatcher_files, mismatcher_path))


def get_char_distribution(text: str) -> dict:
    return Counter(text)


def tokenize_and_tag(bible: dict[int, list[str]], tokenizer, lowercase: bool) -> dict[int, list[list[TaggedWord]]]:
    by_book = defaultdict(list)
    for book_id, verses in bible.items():
        for verse in verses:
            if not verse.strip():
                continue
            if lowercase:
                verse = verse.lower()
            doc = tokenizer(verse)
            tagged_words = [TaggedWord(el.text, el.pos_) for el in doc]
            by_book[book_id].append(tagged_words)
    return by_book


def read_selected_verses(filename: str,
                         lowercase: bool,
                         chosen_books: list[int],
                         tokenizer) -> tuple[dict[int, list[list[TaggedWord]]], dict[int, dict]]:
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Select the books we are interested in
    selected_book_verses = select_samples(bible, chosen_books)
    # Tokenize and get POS tags
    tokenized = tokenize_and_tag(selected_book_verses, tokenizer, lowercase=lowercase)
    # Flatten all characters
    book_chars = {book_id: ''.join([tagged_word.word for verse in verses for tagged_word in verse])
                  for book_id, verses in tokenized.items()}
    # Obtain the repertoire of symbols
    book_char_counter = {book_id: get_char_distribution(chars_in_book) for book_id, chars_in_book in book_chars.items()}
    return tokenized, book_char_counter


def join_words(verse: list[TaggedWord], locations: list[int]) -> list[TaggedWord]:
    """
    Join two words within a verse
    :param verse: a list of tagged words
    :param locations: the locations at which we wish to merge
    :returns: a list of verses consisting of tagged words
    """
    assert all([locations[i] > locations[i + 1] for i in range(len(locations) - 1)])
    location_set = set(locations)
    assert len(location_set) == len(locations)
    joined = []
    i = 0
    while i < len(verse):
        if i in location_set:
            # TODO: we are taking the second POS tag. This should be generalized to other languages
            joined_tagged_word = TaggedWord(verse[i].word + ' ' + verse[i + 1].word, verse[i + 1].pos)
            joined.append(joined_tagged_word)
            i += 2
        else:
            joined.append(verse[i])
            i += 1
    return joined


def merge_positions(verses: list[list[TaggedWord]], positions: list[tuple[int, int]]) -> list[list[TaggedWord]]:
    """
    Merge the noun-noun pairs at the indicated positions
    :param verses: a list of verses, each of which is a list of TaggedWord
    :param positions: a list of positions where the merges should occur
    :returns: a list of verses with the noun-noun pairs merged
    """
    verse_locations = defaultdict(list)
    for position in positions:
        verse_locations[position[0]].append(position[1])
    verse_locations = {verse: sorted(locations, reverse=True) for verse, locations in verse_locations.items()}
    for verse_ix, locations in verse_locations.items():
        verses[verse_ix] = join_words(verses[verse_ix], locations)
    return verses


def replace_top_nnc(verses: list[list[TaggedWord]], pos_tags: list[str]) -> list[list[TaggedWord]]:
    """
    Find the most commonly occurring disjoint noun-noun pair and merge it
    :param verses: the list of POS-tagged verses in the corpus
    :param pos_tags: a tuple containing the POS tags that represent nouns
    :return: the new verses after merging the most common noun-noun pair
    """
    nn_positions = defaultdict(list)
    for j, verse in enumerate(verses):
        for i, tagged_word in enumerate(verse[:-1]):
            if all([el in pos_tags for el in (tagged_word.pos, verse[i + 1].pos)]):
                nn_positions[tagged_word.word + ' ' + verse[i + 1].word].append((j, i))
    # Now the noun-noun bigram with the longest list of positions is the most frequent noun-noun bigram
    if not nn_positions:
        return []
    top_nn = max(nn_positions, key=lambda x: len(nn_positions[x]))
    return merge_positions(verses, nn_positions[top_nn])


def create_word_pasted_sets(id_verses: dict[int, list[list[TaggedWord]]], steps_to_save: set,
                            pos_tags_to_merge: list[str]) -> dict[int, dict[int, list[list[str]]]]:
    max_merges = max(steps_to_save)
    book_id_versions = {}
    for book_id, tokens in id_verses.items():
        joined_verses = {}
        last_version = tokens.copy()
        if 0 in steps_to_save:
            joined_verses[0] = [[tagged_word.word for tagged_word in verse] for verse in tokens]
        for n_joins in range(1, max_merges + 1):
            last_version = replace_top_nnc(last_version, pos_tags_to_merge)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = [[tagged_word.word for tagged_word in verse] for verse in last_version]
        book_id_versions[book_id] = joined_verses
    return book_id_versions


def run_word_pasting(filename: str,
                     lowercase: bool,
                     remove_mismatcher_files: bool,
                     chosen_books: list,
                     merge_steps_to_save: set[int],
                     output_file_path: str,
                     mismatcher_path: str,
                     pos_tags_to_merge: list[str],
                     tokenizer) -> dict:
    selected_book_verses, book_char_counter = read_selected_verses(filename,
                                                                   lowercase,
                                                                   chosen_books,
                                                                   tokenizer)
    book_id_versions = create_word_pasted_sets(selected_book_verses, merge_steps_to_save, pos_tags_to_merge)
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
                                                       book_char_counter[book_id],
                                                       mismatcher_path)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


if __name__ == '__main__':
    assert len(sys.argv) == 6, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]  # The filename of the mismatcher jar
    spacy_model = sys.argv[5]  # The name of the Spacy model for tokenization and POS tagging

    merge_steps = set(list(range(1000)))
    noun_pos_tags = ['NOUN', 'PROPN']
    spacy_tokenizer = spacy.load(spacy_model)
    koplenig_et_al_books = [40, 41, 42, 43, 44, 66]

    book_entropies = run_word_pasting(bible_filename, lowercase=True, remove_mismatcher_files=True,
                                      chosen_books=koplenig_et_al_books, merge_steps_to_save=merge_steps,
                                      output_file_path=temp_dir, mismatcher_path=mismatcher_file,
                                      pos_tags_to_merge=noun_pos_tags, tokenizer=spacy_tokenizer)

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
