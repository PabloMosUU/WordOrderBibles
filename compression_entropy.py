import data
import random
import string


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
    raise NotImplementedError()


def to_file(text: str, orig_filename: str, appendix: str) -> str:
    raise NotImplementedError()


def run_mismatcher(version: str, preprocessed_filename: str) -> list:
    raise NotImplementedError()


def entropy(mismatches: list) -> float:
    raise NotImplementedError()


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
    version_mismatches = {k: run_mismatcher(k, v) for k, v in filenames.items()}
    # Compute the entropy
    version_entropy = {version: entropy(mismatches) \
                       for version, mismatches in version_mismatches.items()}
    return version_entropy
