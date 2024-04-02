import sys
import json
from compression_entropy import read_selected_verses, get_entropies
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer(verses: list, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    # noinspection PyPropertyAccess
    tokenizer.pre_tokenizer = Whitespace()
    # noinspection PyArgumentList
    trainer = BpeTrainer(vocab_size=vocab_size)
    tokenizer.train_from_iterator([' '.join(verse) for verse in verses], trainer)
    return tokenizer


def encode_verses(verse_list: list, tokenizer: Tokenizer) -> list:
    return [tokenizer.encode(' '.join(verse)).tokens.copy() for verse in verse_list]


def build_merge_history(seq_tokens: list, tokenizer: Tokenizer) -> dict:
    raise NotImplementedError()


def create_word_split_sets(id_verses: dict, n_all_merges: int) -> dict:
    """
    Given text from a bible, split it using BPE
    :param id_verses: map book IDs to a (ordered) list of verses in the book, each verse being a list of tokens
    :param n_all_merges: number of merges to train the tokenizer aiming to build the entire merge history
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
        increase_tokens = build_merge_history(verses, full_tokenizer)
        book_id_versions[book_id] = increase_tokens
    return book_id_versions


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
    book_id_versions = create_word_split_sets(selected_book_verses, n_merges)

    book_id_entropies = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            print(n_pairs, end='')
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_v{n_pairs}'
            n_pairs_entropies[n_pairs] = (get_entropies(verse_tokens,
                                                        base_filename,
                                                        remove_mismatcher_files,
                                                        char_counter,
                                                        mismatcher_path),
                                          len(set([el for lis in verse_tokens for el in lis])))
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


def has_completed_merges(orig_verse_tokens: list, trained_bpe_tokenizer: Tokenizer) -> bool:
    orig_verses = [' '.join(el) for el in orig_verse_tokens]
    pre_tokenizer = Whitespace()
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
    n_merges = int(sys.argv[5])    # The number of merges to attempt to reconstruct the entire merge history

    split_steps = set([ell for lis in [list(el) for el in (range(-900, 1000, 100), range(-10000, 11000, 1000))] \
                       for ell in lis])

    book_entropies = {}
    for bid in [40, 41, 42, 43, 44, 66]:
        file_book_entropies = run_word_splitting(bible_filename,
                                                 lowercase=True,
                                                 remove_mismatcher_files=True,
                                                 chosen_books=[bid],
                                                 truncate_books=False,
                                                 n_merges=n_merges,
                                                 output_file_path=temp_dir,
                                                 mismatcher_path=mismatcher_file)
        if bid not in file_book_entropies:
            print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
            continue
        book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
