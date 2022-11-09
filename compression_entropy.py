import data
import random
import string
import os
import numpy as np


def create_random_word(word: str) -> str:
    return ''.join([random.choice(string.ascii_letters) for _ in word])


def mask_word_structure(tokenized: dict) -> dict:
    masked = {}
    word_map = {}
    new_words = set([])
    for verse_id, tokens in tokenized.items():
        masked_tokens = []
        for token in tokens:
            if token not in word_map:
                new_word = create_random_word(token)
                if new_word in new_words:
                    raise ValueError('Random word already exists')
                word_map[token] = new_word
            masked_tokens.append(word_map[token])
        masked[verse_id] = masked_tokens
    return masked


def join_verses(verse_tokens: dict) -> str:
    return ' '.join([' '.join(ell[1]) for ell in sorted(verse_tokens.items(), key=lambda el: el[0])])


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

def get_entropies(verse_tokens: dict, base_filename: str, remove_mismatcher_files: bool) -> dict:
    # Shuffle words within each verse
    shuffled = {verse_id: random.sample(words, k=len(words)) \
                for verse_id, words in verse_tokens.items()}
    # Mask word structure
    masked = mask_word_structure(verse_tokens)
    # Put them in a dictionary
    tokens = {'orig': verse_tokens, 'shuffled': shuffled, 'masked': masked}
    # Join all verses together
    joined = {k: join_verses(v) for k, v in tokens.items()}
    # Save these to files to run the mismatcher
    filenames = {k: to_file(v, base_filename, k) for k, v in joined.items()}
    # Run the mismatcher
    version_mismatches = {version: run_mismatcher(preprocessed_filename, remove_mismatcher_files) \
                          for version, preprocessed_filename in filenames.items()}
    # Compute the entropy
    version_entropy = {version: get_entropy(mismatches) \
                       for version, mismatches in version_mismatches.items()}
    return version_entropy

def run(filename: str, lowercase: bool, remove_mismatcher_files: bool) -> dict:
    """
    Main program to run the entire pipeline on a single bible
    :param filename: the file containing the bible text
    :param lowercase: whether to lowercase the text before processing
    :param remove_mismatcher_files: whether mismatcher files should be deleted after processing
    :return: a dictionary with entropy versions and entropies, keyed by book ID
    """
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Tokenize by splitting on spaces
    tokenized = bible.tokenize(remove_punctuation=False, lowercase=lowercase)
    # Split by book
    _, _, by_book, _, _ = data.join_by_toc(tokenized.verse_tokens)
    book_verse_tokens = {book_id: {str(i): tokens for i, tokens in enumerate(token_list)} \
                         for book_id, token_list in by_book.items()}
    book_base_filename = {book_id: 'output/KoplenigEtAl/' + filename.split('/')[-1] + f'_{book_id}' \
                          for book_id in book_verse_tokens.keys()}
    return {book_id: get_entropies(verse_tokens, book_base_filename[book_id], remove_mismatcher_files) \
            for book_id, verse_tokens in book_verse_tokens.items()}

if __name__ == '__main__':
    print(run('/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/eng-x-bible-world.txt',
              True, True))
