import sys
import re
from collections.abc import MutableMapping
from typing import Iterator


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
    pass


class TokenizedBible:
    def __init__(self, language: str, filename: str, verse_tokens: dict):
        self.language = language
        self.filename = filename
        self.verse_tokens = verse_tokens

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(f'# ISO:\t{self.language}\n')
            f.write(f'# Original filename:\t{self.filename}\n')
            for verse_number, verse_tokens in self.verse_tokens.items():
                f.write(f'{verse_number}\t{" ".join(verse_tokens)}\n')

    def read(self, filename: str):
        raise NotImplementedError()

    def split(self) -> SplitData:
        # Todo: challenge 2
        raise NotImplementedError()


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
