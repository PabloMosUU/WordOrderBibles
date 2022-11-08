import data
import random
import string
import os
import numpy as np


def create_random_word(word: str) -> str:
    return ''.join([random.choice(string.ascii_letters) for _ in word])


def mask_word_structure(tokenized: data.TokenizedBible) -> dict:
    masked = {}
    word_map = {}
    new_words = set([])
    for verse_id, tokens in tokenized.verse_tokens.items():
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


def to_file(text: str, orig_filename: str, appendix: str) -> str:
    dot_parts = orig_filename.split('/')[-1].split('.')
    extension = dot_parts[-1]
    prefix = '.'.join(dot_parts[:-1])
    new_filename = prefix + '_' + appendix + '.' + extension
    with open(new_filename, 'w') as f:
        f.write(text)
    return new_filename


def run_mismatcher(preprocessed_filename: str) -> list:
    mismatcher_filename = preprocessed_filename + '_mismatcher'
    os.system(f"""java -Xmx3500M -jar \
    /home/pablo/ownCloud/WordOrderBibles/Literature/ThirdRound/dataverse_files/shortestmismatcher.jar \
    {preprocessed_filename} {mismatcher_filename}""")
    with open(mismatcher_filename, 'r') as f:
        lines = f.readlines()
    os.remove(mismatcher_filename)
    return parse_mismatcher_lines(lines)


def parse_mismatcher_lines(lines: list) -> list:
    return [int(line.split('\t')[-1].strip()) for line in lines if line != '\n']

def get_entropy(mismatches: list) -> float:
    return 1 / (sum([el/np.log2(i + 2) for i, el in enumerate(mismatches[1:])]) / len(mismatches))


def run(filename: str) -> dict:
    # Read the complete bible
    bible = data.parse_pbc_bible(filename)
    # Tokenize by splitting on spaces
    tokenized = bible.tokenize(remove_punctuation=False, lowercase=False)
    # Shuffle words within each verse
    shuffled = {verse_id: random.sample(words, k=len(words)) \
                for verse_id, words in tokenized.verse_tokens.items()}
    # Mask word structure
    masked = mask_word_structure(tokenized)
    # Put them in a dictionary
    tokens = {'orig': tokenized.verse_tokens, 'shuffled': shuffled, 'masked': masked}
    # Join all verses together
    joined = {k: join_verses(v) for k, v in tokens.items()}
    # Save these to files to run the mismatcher
    filenames = {k: to_file(v, filename, k) for k, v in joined.items()}
    # Run the mismatcher
    version_mismatches = {version: run_mismatcher(preprocessed_filename) \
                          for version, preprocessed_filename in filenames.items()}
    # Compute the entropy
    version_entropy = {version: get_entropy(mismatches) \
                       for version, mismatches in version_mismatches.items()}
    return version_entropy


if __name__ == '__main__':
    run('eng-x-bible-world_sample.txt')
