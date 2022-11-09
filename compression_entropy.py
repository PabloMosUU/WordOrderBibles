import data
import random
import os
import numpy as np


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


def join_verses(verse_tokens: list) -> str:
    """
    Join the verses contained in a list of lists of tokens
    :param verse_tokens: the list of verses, each of which is a list of tokens; order matters
    :return: the concatenated string consisting of all the tokens in the original order
    """
    return ' '.join([' '.join(ell) for ell in verse_tokens])


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

def get_entropies(verse_tokens: list, base_filename: str, remove_mismatcher_files: bool, char_set: str) -> dict:
    # Shuffle words within each verse
    shuffled = [random.sample(words, k=len(words)) for words in verse_tokens]
    # Mask word structure
    masked = mask_word_structure(verse_tokens, char_set)
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

def run(filename: str, lowercase: bool, remove_mismatcher_files: bool, chosen_books: list) -> dict:
    """
    Main program to run the entire pipeline on a single bible
    :param filename: the file containing the bible text
    :param lowercase: whether to lowercase the text before processing
    :param remove_mismatcher_files: whether mismatcher files should be deleted after processing
    :param chosen_books: the books for which you want to compute the entropy (PBC IDs)
    :return: a dictionary with entropy versions and entropies, keyed by book ID
    """
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Tokenize by splitting on spaces
    tokenized = bible.tokenize(remove_punctuation=False, lowercase=lowercase)
    # Obtain the repertoire of symbols
    char_set = ''.join(set(''.join([el for lis in tokenized.verse_tokens.values() for el in lis])))
    # Split by book
    _, _, by_book, _, _ = data.join_by_toc(tokenized.verse_tokens)
    book_base_filename = {book_id: 'output/KoplenigEtAl/' + filename.split('/')[-1] + f'_{book_id}' \
                          for book_id in by_book.keys()}
    return {book_id: get_entropies(by_book[book_id],
                                   book_base_filename[book_id],
                                   remove_mismatcher_files,
                                   char_set) \
            for book_id in chosen_books}

if __name__ == '__main__':
    print(run('/home/pablo/Documents/GitHubRepos/paralleltext/bibles/corpus/eng-x-bible-world.txt',
              lowercase=True, remove_mismatcher_files=True, chosen_books=[40, 41, 42, 43, 44, 66]))
