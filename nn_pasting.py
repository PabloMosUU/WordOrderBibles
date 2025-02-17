from collections import defaultdict

import pandas as pd
import spacy

import data
import sys
import compression_entropy as wp


class TaggedWord:
    def __init__(self, word: str, pos: str):
        self.word = word
        self.pos = pos

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos

    def __repr__(self):
        return f'({self.word}, {self.pos})'


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
    lengths = {sample_id: wp.get_text_length(verses)
               for sample_id, verses in by_book.items()}
    if len(lengths) == 0:
        return {}
    return by_book


def tokenize_and_tag(bible: dict[int, list[str]], tokenizer, lowercase: bool) -> dict[int, list[list[TaggedWord]]]:
    """
    Apply the tokenizer pipeline to this bible, including part-of-speech tagging
    :param bible: the entire content of a bible, in the form of a dictionary mapping book IDs to lists of verses
    :param tokenizer: a Spacy tokenizer
    :param lowercase: whether we want the final result in lowercase
    :returns a dictionary mapping book IDs to a list of lists of POS-tagged words
    """
    by_book = defaultdict(list)
    for book_id, verses in bible.items():
        for verse in verses:
            if not verse.strip():
                continue
            doc = tokenizer(verse)
            tagged_words = [TaggedWord(el.text.lower() if lowercase else el.text, el.pos_) for el in doc]
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
    # TODO: the bit below is completely independent and should be in a separate function
    # Flatten all characters
    book_chars = {book_id: ''.join([tagged_word.word for verse in verses for tagged_word in verse])
                  for book_id, verses in tokenized.items()}
    # Obtain the repertoire of symbols
    book_char_counter = {book_id: wp.get_char_distribution(chars_in_book) for book_id, chars_in_book in
                         book_chars.items()}
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


def replace_top_nnc(verses: list[list[TaggedWord]],
                    pos_tags: list[str]) -> tuple[list[list[TaggedWord]], str]:
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
        return [], ''
    top_nn = max(nn_positions, key=lambda x: len(nn_positions[x]))
    return merge_positions(verses, nn_positions[top_nn]), top_nn


def create_word_pasted_sets(id_verses: dict[int, list[list[TaggedWord]]], steps_to_save: set,
                            pos_tags_to_merge: list[str]) -> dict[int, dict[int, tuple[list[list[str]], str]]]:
    max_merges = max(steps_to_save)
    book_id_versions = {}
    for book_id, tokens in id_verses.items():
        joined_verses = {}
        last_version = tokens.copy()
        if 0 in steps_to_save:
            joined_verses[0] = ([[tagged_word.word for tagged_word in verse] for verse in tokens], '')
        for n_joins in range(1, max_merges + 1):
            last_version, merged_pair = replace_top_nnc(last_version, pos_tags_to_merge)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = ([[tagged_word.word for tagged_word in verse] for verse in last_version],
                                          merged_pair)
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
                     tokenizer) -> dict[int, dict[int, tuple[dict, str]]]:
    selected_book_verses, book_char_counter = read_selected_verses(filename,
                                                                   lowercase,
                                                                   chosen_books,
                                                                   tokenizer)
    book_id_versions = create_word_pasted_sets(selected_book_verses, merge_steps_to_save, pos_tags_to_merge)
    book_id_entropies_and_merge_pairs = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies_and_merge_pair = {}
        for n_pairs, verse_tokens_and_merge_pairs in n_pairs_verses.items():
            verse_tokens = verse_tokens_and_merge_pairs[0]
            merge_pair = verse_tokens_and_merge_pairs[1]
            print(n_pairs, end='')
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_v{n_pairs}'
            n_pairs_entropies_and_merge_pair[n_pairs] = (wp.get_entropies(verse_tokens,
                                                                          base_filename,
                                                                          remove_mismatcher_files,
                                                                          book_char_counter[book_id],
                                                                          mismatcher_path),
                                                         merge_pair)
        book_id_entropies_and_merge_pairs[book_id] = n_pairs_entropies_and_merge_pair
    return book_id_entropies_and_merge_pairs


if __name__ == '__main__':
    assert len(sys.argv) == 6, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]  # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]  # The filename of the mismatcher jar
    spacy_model = sys.argv[5]  # The name of the Spacy model for tokenization and POS tagging

    merge_steps = set(list(range(1000)))
    noun_pos_tags = ['NOUN']
    spacy_tokenizer = spacy.load(spacy_model)
    koplenig_et_al_books = [40, 41, 42, 43, 44, 66]

    book_entropies = run_word_pasting(bible_filename, lowercase=True, remove_mismatcher_files=True,
                                      chosen_books=koplenig_et_al_books, merge_steps_to_save=merge_steps,
                                      output_file_path=temp_dir, mismatcher_path=mismatcher_file,
                                      pos_tags_to_merge=noun_pos_tags, tokenizer=spacy_tokenizer)

    book_ids = []
    n_merges = []
    text_versions = []
    entropies = []
    merged_pairs = []
    for bid, n_merges_entropies_and_merged_pairs in book_entropies.items():
        for nm, entropies_and_merged_pairs in n_merges_entropies_and_merged_pairs.items():
            ee = entropies_and_merged_pairs[0]
            mp = entropies_and_merged_pairs[1]
            for version, entropy in ee.items():
                book_ids.append(bid)
                n_merges.append(nm)
                text_versions.append(version)
                entropies.append(entropy)
                merged_pairs.append(mp)
    df = pd.DataFrame({'book_id': book_ids, 'n_merges': n_merges, 'text_version': text_versions, 'entropy': entropies,
                       'merged_pair': merged_pairs, 'filename': bible_filename})
    df.to_csv(output_filename)
