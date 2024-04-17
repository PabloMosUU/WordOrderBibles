import copy
import os
import sys
import json
import random
from compression_entropy import read_selected_verses, run_mismatcher, get_entropy, to_file, create_random_word
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer

from util import Token


def mask_word_structure(tokenized: list, char_str: str, char_weights: list) -> list:
    masked = []
    word_map = {}
    new_words = set([])
    for tokens in tokenized:
        masked_tokens = []
        for token_obj in tokens:
            token = token_obj.token
            if token not in word_map:
                new_word = create_random_word(len(token), char_str, char_weights)
                if new_word in new_words:
                    raise ValueError('Random word already exists')
                word_map[token] = new_word
            masked_tokens.append(Token(word_map[token], token_obj.is_start_of_word))
        masked.append(masked_tokens)
    return masked


def train_tokenizer(verses: list, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    # noinspection PyPropertyAccess
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # noinspection PyArgumentList
    trainer = BpeTrainer(vocab_size=vocab_size)
    tokenizer.train_from_iterator([' '.join(verse) for verse in verses], trainer)
    return tokenizer


def encode_verses(verse_list: list, tokenizer: Tokenizer) -> list:
    return [tokenizer.encode(' '.join(verse)).tokens.copy() for verse in verse_list]


def get_merge_steps(merge_list_file: str) -> list:
    with open(merge_list_file) as f:
        lines = f.readlines()
    assert lines[0].startswith('#') and not lines[1].startswith('#')
    merge_steps = [line.strip().split(' ') for line in lines[1:]]
    for i, line in enumerate(merge_steps):
        if len(line) != 2 or line[0] != line[0].strip() or line[1] != line[1].strip():
            print(i, line, type(line))
            raise ValueError()
    return merge_steps


def split_chars(verse_tokens: list) -> list:
    return [[list(token) for token in tokens] for tokens in verse_tokens]


def apply_merge(seq_token_sub_tokens: list, merge_step: list) -> list:
    for i, verse in enumerate(seq_token_sub_tokens):
        for j in range(len(verse)):
            token = verse[j]
            parts = []
            k = 0
            while k < len(token):
                if k == len(token) - 1:
                    parts.append(token[k])
                    k += 1
                elif token[k] == merge_step[0] and token[k + 1] == merge_step[1]:
                    parts.append(token[k] + token[k + 1])
                    k += 2
                else:
                    parts.append(token[k])
                    k += 1
            verse[j] = parts
    return seq_token_sub_tokens


def build_merge_history(seq_tokens: list, merge_steps: list) -> dict:
    seq_token_sub_tokens = split_chars(seq_tokens)
    merge_history = {len(merge_steps): copy.deepcopy(seq_token_sub_tokens)}
    save_step = int(len(merge_steps) / 10 + 0.5)
    for i, merge_step in enumerate(merge_steps):
        seq_token_sub_tokens = apply_merge(seq_token_sub_tokens, merge_step)
        n_merges_so_far = i + 1
        n_splits_so_far = len(merge_steps) - n_merges_so_far
        if len(merge_steps) < 10 or n_splits_so_far == 0 or n_splits_so_far % save_step == 0:
            merge_history[n_splits_so_far] = copy.deepcopy(seq_token_sub_tokens)
    return merge_history


def flatten_sequences(splits_sequences: dict) -> dict:
    flattened = {}
    for splits, sequences in splits_sequences.items():
        flattened_sequences = []
        for seq in sequences:
            flattened_tokens = []
            for token in seq:
                # TODO: this should really be done at the level of reading the bibles
                if len(token) == 0:
                    continue
                flattened_tokens.append(Token(token[0], True))
                for sub_token in token[1:]:
                    flattened_tokens.append(Token(sub_token, False))
            flattened_sequences.append(flattened_tokens)
        flattened[splits] = flattened_sequences
    return flattened


def get_merge_history(seq_tokens: list, tokenizer: Tokenizer, temp_path: str, unique_file_id: str) -> dict:
    # Save the tokenizer to a file
    tokenizer_files = tokenizer.model.save(temp_path, f'merge_history_tokenizer_{unique_file_id}')
    merges_filename = tokenizer_files[1]
    # Use get_merge_steps to retrieve the list of merge steps
    merge_steps = get_merge_steps(merges_filename)
    # Delete the temporary file
    for file in tokenizer_files:
        os.remove(file)
    # For each n_merges/10, use apply_merge iteratively to construct a stage of reconstruction
    merge_history = build_merge_history(seq_tokens, merge_steps)
    # Flatten the sequences
    splits_tokens = flatten_sequences(merge_history)
    return splits_tokens


def create_word_split_sets(id_verses: dict, n_all_merges: int, temp_path: str, unique_file_id: str) -> dict:
    """
    Given text from a bible, split it using BPE
    :param id_verses: map book IDs to a (ordered) list of verses in the book, each verse being a list of tokens
    :param n_all_merges: number of merges to train the tokenizer aiming to build the entire merge history
    :param temp_path: a path to a directory where temporary files can be saved
    :param unique_file_id: a unique identifier for the file from which id_verses was created
    :return: dictionary mapping book IDs to (dictionary mapping number of splits to list of tokens)
    """
    book_id_versions = {}
    for book_id, verses in id_verses.items():
        tokens = [el for lis in verses for el in lis]
        alphabet_size = len(set(list(' '.join(tokens))))
        # Train tokenizer for a large number of merges, aiming to reconstruct the entire merge history
        full_tokenizer = train_tokenizer(verses, alphabet_size + n_all_merges)
        # verify that all merges have been completed
        if not has_completed_merges(verses, full_tokenizer):
            raise NotImplementedError('merge history incomplete')
        # Iteratively use merge history up to a point to encode verses
        increase_tokens = get_merge_history(verses, full_tokenizer, temp_path, unique_file_id)
        book_id_versions[book_id] = increase_tokens
    return book_id_versions


def join_verses(verse_tokens: list, insert_spaces: bool) -> str:
    """
    Join the verses contained in a list of lists of tokens
    :param verse_tokens: the list of verses, each of which is a list of tokens; order matters
    :param insert_spaces: whether we want to insert spaces between the tokens
    :return: the concatenated string consisting of all the tokens in the original order
    """
    sep = ' ' if insert_spaces else ''
    verses = []
    for verse in verse_tokens:
        verse_text = ''
        for j, token in enumerate(verse):
            if j != 0 and token.is_start_of_word:
                verse_text += sep
            verse_text += token.token
        verses.append(verse_text)
    return sep.join(verses)


# TODO: deduplicate this function and compression_entropy.get_entropies by de-duplicating join_verses
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


def get_output_file_dir(output_file_path: str, filename: str) -> str:
    output_file_dir = os.path.join(
        output_file_path,
        os.path.splitext(os.path.basename(filename))[0].replace('.', '_')
    )
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)
    return output_file_dir


def run_word_splitting(filename: str,
                       lowercase: bool,
                       remove_mismatcher_files: bool,
                       chosen_books: list,
                       truncate_books: bool,
                       n_merges: int,
                       output_file_path: str,
                       mismatcher_path: str) -> dict:
    selected_book_verses, char_counter = read_selected_verses(filename,
                                                              lowercase,
                                                              chosen_books,
                                                              truncate_books)
    # Create the split versions using BPE
    book_id_versions = create_word_split_sets(selected_book_verses, n_merges, output_file_path, filename.split('/')[-1])

    book_id_entropies = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            print(n_pairs, end='')
            base_dir = get_output_file_dir(output_file_path, filename)
            base_filename = os.path.join(base_dir, f'{os.path.basename(filename)}_{book_id}_v{n_pairs}')
            n_pairs_entropies[n_pairs] = get_entropies(verse_tokens,
                                                       base_filename,
                                                       remove_mismatcher_files,
                                                       char_counter,
                                                       mismatcher_path)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


def has_completed_merges(orig_verse_tokens: list, trained_bpe_tokenizer: Tokenizer) -> bool:
    orig_verses = [' '.join(el) for el in orig_verse_tokens]
    pre_tokenizer = WhitespaceSplit()
    orig_verses_pre_tokenized = [[ell[0] for ell in pre_tokenizer.pre_tokenize_str(el)] for el in orig_verses]
    encoded_verse_tokens = encode_verses(orig_verses_pre_tokenized, trained_bpe_tokenizer)
    for verse_ix, verse_tokens in enumerate(encoded_verse_tokens):
        for token_ix, token in enumerate(verse_tokens):
            if orig_verses_pre_tokenized[verse_ix][token_ix] != token:
                print('DEBUG: Different', orig_verses_pre_tokenized[verse_ix][token_ix], token)
                return False
    return True


if __name__ == '__main__':
    assert len(sys.argv) == 6, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename n_merges_full'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]  # The filename of the mismatcher jar
    n_merges_full = int(sys.argv[5])  # The number of merges to attempt to reconstruct the entire merge history

    book_entropies = {}
    for bid in [40, 41, 42, 43, 44, 66]:
        file_book_entropies = run_word_splitting(bible_filename,
                                                 lowercase=True,
                                                 remove_mismatcher_files=True,
                                                 chosen_books=[bid],
                                                 truncate_books=False,
                                                 n_merges=n_merges_full,
                                                 output_file_path=temp_dir,
                                                 mismatcher_path=mismatcher_file)
        if bid not in file_book_entropies:
            print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
            continue
        book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
    print('INFO: completed successfully')
