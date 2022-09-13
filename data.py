import random
import sys
import re
from collections.abc import MutableMapping
from typing import Iterator

import numpy as np
from sklearn.model_selection import train_test_split

START_OF_VERSE_TOKEN = '<SOS>'
END_OF_VERSE_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
CHUNK_END_TOKEN = '<END>'

class IndelibleDict(MutableMapping):
    def __init__(self):
        self.store = dict()

    def __setitem__(self, k, v) -> None:
        if k in self.store:
            raise ValueError(f'Key {k} already present in dictionary')
        self.store.__setitem__(k, v)

    def __delitem__(self, k) -> None:
        self.store.__delitem__(k)

    def __getitem__(self, k):
        return self.store.__getitem__(k)

    def __len__(self) -> int:
        return self.store.__len__()

    def __iter__(self) -> Iterator:
        return self.store.__iter__()


class SplitData:
    def __init__(self, train_data, hold_out_data, test_data):
        self.train_data = train_data
        self.hold_out_data = hold_out_data
        self.test_data = test_data
        self.train_word_to_ix = self._word_to_ix()
        self.train_ix_to_word = {v:k for k,v in self.train_word_to_ix.items()}

    def get(self, partition: str) -> list:
        if partition == 'train':
            return self.train_data
        elif partition == 'holdout':
            return self.hold_out_data
        elif partition == 'test':
            return self.test_data
        else:
            raise ValueError(f'Unknown partition {partition}')

    def shuffle_chop(self, partition: str, sequence_length: int) -> list:
        """
        Shuffles the verses and chunks the outcome in sequences of fixed length
        :param partition: the partition (train/holdout/test) you want to work on
        :param sequence_length: the length of the chopped sequences
        :return: a list of sequences, each of which is a list of tokens
        """
        verses = self.get(partition)
        random.shuffle(verses)
        verses = [el + [END_OF_VERSE_TOKEN] for el in verses]
        flattened = [el for lis in verses for el in lis]
        flattened += ((sequence_length - (len(flattened) % sequence_length)) * [PAD_TOKEN])
        chunks = [flattened[i:i+sequence_length] for i in range(0, len(flattened), sequence_length)]
        chunks = [chunk for chunk in chunks if PAD_TOKEN not in chunk]
        return chunks

    def _word_to_ix(self):
        word_to_ix = {}
        # For each words-list (sentence) and tags-list in each tuple of training_data
        for sent in self.train_data:
            for word in sent:
                if word not in word_to_ix:  # word has not been assigned an index yet
                    word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
        for special_token in (START_OF_VERSE_TOKEN, END_OF_VERSE_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN, CHUNK_END_TOKEN):
            if special_token not in word_to_ix:
                word_to_ix[special_token] = len(word_to_ix)
        return word_to_ix

class TokenizedBible:
    def __init__(self, language: str, filename: str, verse_tokens: dict):
        """
        Create a tokenized bible that is ready for data splitting and model training
        :param language: the ISO code for the language
        :param filename: the filename of the original (non-tokenized) data
        :param verse_tokens: a dictionary mapping verse numbers or any other unique ID to tokens
        """
        self.language = language
        self.filename = filename
        self.verse_tokens = verse_tokens

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(f'# ISO:\t{self.language}\n')
            f.write(f'# Original filename:\t{self.filename}\n')
            for verse_number, verse_tokens in self.verse_tokens.items():
                f.write(f'{verse_number}\t{" ".join(verse_tokens)}\n')

    @staticmethod
    def read(filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        language = lines[0].split('\t')[1].strip()
        original_filename = lines[1].split('\t')[1].strip()
        verse_tokens = {el.split('\t')[0]: el.split('\t')[1].strip() \
                        for el in lines[2:]}
        return TokenizedBible(language, original_filename, verse_tokens)

    def split(self, hold_out_fraction: float, test_fraction: float) -> SplitData:
        """
        Split the data into a training set, a holdout set, and test set
        :param hold_out_fraction: the fraction of all data that should go into holdout
        :param test_fraction: the fraction of all data that should go into test
        :return: an object containing the split data
        """
        train_and_hold_out_data, test_data = train_test_split(
            list(self.verse_tokens.values()),
            test_size = test_fraction if test_fraction > 0 else None
        )
        hold_out_fraction_of_dev = hold_out_fraction / (1 - test_fraction)
        train_data, hold_out_data = train_test_split(
            train_and_hold_out_data,
            test_size = hold_out_fraction_of_dev
        )
        return SplitData(train_data, hold_out_data, test_data)


class Bible:
    """
    This class is mutable because I don't want to consume too much memory
    """
    def __init__(self, language: str, filename: str):
        self.language = language
        self.filename = filename

    def tokenize(self, remove_punctuation: bool, lowercase: bool) -> TokenizedBible:
        raise NotImplementedError()

class PbcBible(Bible):
    def __init__(self, language: str, filename: str, content: IndelibleDict, hidden_content: IndelibleDict):
        Bible.__init__(self, language, filename)
        self.content = content
        self.hidden_content = hidden_content

    def tokenize(self, remove_punctuation: bool, lowercase: bool) -> TokenizedBible:
        verse_tokens = {}
        for verse, text in self.content.items():
            if text.strip() == '':
                continue
            verse_tokens[verse] = tokenize(text, remove_punctuation, lowercase)
        return TokenizedBible(self.language, self.filename, verse_tokens)

    @staticmethod
    def to_dictionaries(comment_lines: list, content_lines: list):
        comments = IndelibleDict()
        for key, value in comment_lines:
            comments[key] = value
        content, hidden_content = IndelibleDict(), IndelibleDict()
        for key, value, is_commented in content_lines:
            if is_commented:
                hidden_content[key] = value
            else:
                content[key] = value
        return comments, content, hidden_content

def tokenize(text: str, remove_punctuation: bool, lowercase: bool) -> list:
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        tokens = re.findall('(\\S*\\w\\S*) ?', text)
    else:
        tokens = text.split(' ')
    return tokens

def parse_pbc_bible_lines(lines: list, parse_content: bool, filename: str) -> PbcBible:
    # Assume that the file starts with comments, and then it moves on to content
    # The comments have alpha keys that start with a hash and end in colon
    # The content can optionally be commented out
    in_comments = True
    comment_lines, content_lines = [], []
    content_pattern = '#? ?(\\d{1,8}) ?\t(.*)\\s*'
    for line in lines:
        if in_comments:
            comment_match = re.fullmatch('# ([\\w\\d-]+):\\s+(.*)\\s*', line)
            if comment_match:
                comment_lines.append((comment_match.group(1), comment_match.group(2)))
            else:
                content_match = re.fullmatch(content_pattern, line)
                if content_match:
                    if not parse_content:
                        break
                    content_lines.append((content_match.group(1), content_match.group(2), line[0] == '#'))
                    in_comments = False
                else:
                    comment_lines[-1] = (comment_lines[-1][0], comment_lines[-1][1] + '\n' + line)
        else:
            content_match = re.fullmatch(content_pattern, line)
            if content_match:
                content_lines.append((content_match.group(1), content_match.group(2), line[0] == '#'))
            else:
                raise Exception(f'{line} does not match an expected format')
    comments, content, hidden_content = PbcBible.to_dictionaries(comment_lines, content_lines)
    language = comments['closest_ISO_639-3']
    return PbcBible(language, filename, content, hidden_content)

def parse_pbc_bible(filename: str) -> PbcBible:
    with open(filename) as f:
        lines = f.readlines()
    return parse_pbc_bible_lines(lines, parse_content=True, filename=filename)


def parse_file(filename: str, corpus: str) -> Bible:
    if corpus.lower() == 'pbc':
        return parse_pbc_bible(filename)
    else:
        raise NotImplementedError()


def preprocess(bible: Bible) -> TokenizedBible:
    """
    Data preprocessing for bibles
    :param bible: the bible you want to preprocess
    :return: the preprocessed bible
    """
    # See lowercasing caveats in Bentz et al Appendix A.1
    return bible.tokenize(remove_punctuation=True, lowercase=True)

def process_bible(filename: str, corpus: str) -> TokenizedBible:
    structured_bible = parse_file(filename, corpus)
    return preprocess(structured_bible)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'USAGE: {sys.argv[0]} <corpus> <bible_filename> <output_filename>')
        exit(-1)
    bible_corpus = sys.argv[1]
    bible_filename = sys.argv[2]
    output_filename = sys.argv[3]
    pre_processed_bible = process_bible(bible_filename, bible_corpus)
    pre_processed_bible.save(output_filename)


def prepare_sequence(seq: list, to_ix: dict) -> list:
    return [to_ix[w] if w in to_ix else to_ix[UNKNOWN_TOKEN] for w in seq]


def batch(dataset: list, batch_size: int, word_index: dict) -> tuple:
    """
    Breaks up a dataset into batches and puts them in tensor format for PyTorch to train
    :param dataset: a list of sequences, each of which is a list of tokens
    :param batch_size: the desired batch size
    :param word_index: a map from words to indices
    :return: a tensor containing the entire dataset separated into batches, with appropriate padding
    """
    # Break up into batches
    batches = [dataset[batch_size*i:batch_size*(i+1)] for i in range(int(np.ceil(len(dataset)/batch_size)))]

    # Sort sequences in each batch from longest to shortest
    sorted_batches = [sorted(b, key=lambda seq: -len(seq)) for b in batches]

    # Add start- and end-of-sentence tokens
    enclosed = [[[START_OF_VERSE_TOKEN] + seq + [END_OF_VERSE_TOKEN] for seq in b] for b in sorted_batches]
    original_sequence_lengths = [[len(seq) for seq in b] for b in enclosed]

    # Pad inside each batch using a padding token
    padded_batches = [pad_batch(b) for b in enclosed]

    # Convert words to indices
    as_indices = [[[word_index[w] if w in word_index else word_index[UNKNOWN_TOKEN] for w in seq] for seq in b] \
                  for b in padded_batches]

    return as_indices, original_sequence_lengths


def pad_batch(sequences: list) -> list:
    """
    Given a list of sequences, pad all but one to have the same length as the longest one
    :param sequences: a list of sequences
    :return: the same sequences with a padding symbol added accordingly
    """
    max_length = max([len(el) for el in sequences])
    padded = [seq + [PAD_TOKEN] * (max_length - len(seq)) for seq in sequences]
    return padded
