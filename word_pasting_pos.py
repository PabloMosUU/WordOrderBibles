from collections import defaultdict

import json
import sys
import spacy

import compression_entropy as ce
import word_pasting as wp


def pos_tagging(wordlist: list, language_model: str) -> list:
    """
    Uses spacy POS tagger to return POS tags of the words in the verses.
    """
    nlp = spacy.load(language_model)
    tagged_output = []

    for sublist in wordlist:
        text_string = ' '.join(sublist)
        doc = nlp(text_string)
        tagged_sublist = [(token.text, token.tag_) for token in doc]
        tagged_output.append(tagged_sublist)

    return tagged_output


def merge_positions_pos(verses: list, bigram_tag: tuple) -> dict:
    """
    Merges the word pairs that have the POS tag combination.
    """
    (first_tag, second_tag) = bigram_tag

    result = []
    for verse in verses:
        merged = []
        i = 0
        while i < len(verse):
            word, tag = verse[i]
            if tag == first_tag and i + 1 < len(verse) and verse[i + 1][1] == second_tag:
                next_word, next_tag = verse[i + 1]
                merged.append((word + next_word, tag + next_tag))
                i += 2  
            else:
                merged.append((word, tag))
                i += 1
        result.append(merged)

    return result


def replace_top_bigram_pos(verses: list) -> list:
    """
    Finds the POS-tags of the most frequent occuring word pair.
    """
    bigram_positions = defaultdict(list)
    for j, verse in enumerate(verses):
        for i, word in enumerate(verse[:-1]):
            bigram_positions[(word, verse[i + 1])].append((j, i))
    if not bigram_positions:
        return []
    ((_, tag1),(_, tag2)) = max(bigram_positions, key=lambda x: len(bigram_positions[x]))
    top_bigram_tags = (tag1, tag2)

    return merge_positions_pos(verses, top_bigram_tags)


def create_word_pasted_sets_pos(id_verses: dict, steps_to_save: set, pos_model: str) -> dict:
    """
    Create versions of the Bible books with the different number of merges.
    """
    max_merges = max(steps_to_save)
    book_id_versions = {}

    for book_id, tokens in id_verses.items():
        last_version = pos_tagging(tokens, pos_model)
        joined_verses = {}
        if 0 in steps_to_save:
            joined_verses[0] = tokens.copy()
        for n_joins in range(1, max_merges + 1):
            last_version = replace_top_bigram_pos(last_version)
            if not last_version:
                break
            if n_joins in steps_to_save:
                joined_verses[n_joins] = [[word for word, _ in verse] for verse in last_version]
        book_id_versions[book_id] = joined_verses

    return book_id_versions


def create_paste_files(verses: list, filename: str):
    """
    Extra function to nicely print pasted versions.
    """
    with open(filename, "w", encoding="utf-8") as f:
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
                     pos_model: str) -> dict:
    """
    General function for running the word pasting. Creates pasted versions and calculated entropy
    of these versions.
    """
    if chosen_books == ['full_bible']:
        return (run_word_pasting_all(filename, lowercase, remove_mismatcher_files, truncate_books, merge_steps_to_save, output_file_path, mismatcher_path, pos_model))

    selected_book_verses, char_counter = ce.read_selected_verses(filename,
                                                                 lowercase,
                                                                 chosen_books,
                                                                 truncate_books)
    book_id_versions = create_word_pasted_sets_pos(selected_book_verses, merge_steps_to_save, pos_model)
    book_id_entropies = {}

    for book_id, n_pairs_verses in book_id_versions.items():
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            #create_paste_files(verse_tokens, "text.txt") #Uncomment when merged files are wanted to be saved
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_pos_v{n_pairs}'
            n_pairs_entropies[n_pairs] = wp.get_entropies(verse_tokens,
                                                       base_filename,
                                                       remove_mismatcher_files,
                                                       char_counter,
                                                       mismatcher_path)
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


def run_word_pasting_all(filename: str,
                         lowercase: bool,
                         remove_mismatcher_files: bool,
                         truncate_books: bool,
                         merge_steps_to_save: set,
                         output_file_path: str,
                         mismatcher_path: str,
                         pos_model: str) -> dict:

    all_verses_dict, char_counter = ce.read_all_verses(filename, lowercase, truncate_books)

    verse_list = all_verses_dict['full_bible']

    n_pairs_versions = create_word_pasted_sets_pos({'full_bible': verse_list}, merge_steps_to_save, pos_model)['full_bible']

    n_pairs_entropies = {}
    for n_pairs, verse_tokens in n_pairs_versions.items():
        base_filename = f'{output_file_path}/{filename.split("/")[-1]}_fullbibles_v{n_pairs}'
        n_pairs_entropies[n_pairs] = wp.get_entropies(
            verse_tokens,
            base_filename,
            remove_mismatcher_files,
            char_counter,
            mismatcher_path
        )
    return {'full_bible': n_pairs_entropies}


if __name__ == '__main__':
    assert len(sys.argv) == 6, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename pos_model_name'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]        # The directory where Mismatcher files are saved
    output_filename = sys.argv[3] # The filename where entropies will be saved
    mismatcher_file = sys.argv[4] # The filename of the mismatcher jar
    language_model = sys.argv[5]  # The language model used for pos-tagging

    merge_steps = set(range(0, 21))

    book_entropies = {}   
    for bid in [40, 41, 42, 43, 44, 66, 'full_bible']:
        file_book_entropies = run_word_pasting(bible_filename,
                                               lowercase=True,
                                               remove_mismatcher_files=True,
                                               chosen_books=[bid],
                                               truncate_books=False,
                                               merge_steps_to_save=merge_steps,
                                               output_file_path=temp_dir,
                                               mismatcher_path=mismatcher_file,
                                               pos_model = language_model)
        if bid not in file_book_entropies:
            print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
            continue
        book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
