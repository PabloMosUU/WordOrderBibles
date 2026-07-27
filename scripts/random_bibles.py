"""Create bibles consisting of random content, not similar to language

Usage: python random_bibles.py [INPUT_FILE] [OUTPUT_DIR]
Dependencies: none
Author: Pablo Mosteiro
Status: Final
"""
import sys
import collections
import random
import string
from wordorderbibles import data
import os

UNIFORM_ALPHABET = ''.join(set(string.ascii_letters.lower())) + " "  # A-Z, a-z, and space

def split_line_and_newline(line):
    """Safely separate line content from its newline character(s)."""
    if line.endswith('\r\n'):
        return line[:-2], '\r\n'
    if line.endswith('\n') or line.endswith('\r'):
        return line[:-1], line[-1]
    return line, ''

def write_output(comments: dict[str, str], content: dict[str, str], path: str) -> None:
    with open(path, 'w') as f:
        for key, value in comments.items():
            f.write('# ' + key + ':\t' + value + '\n')
        for verse_id in sorted(list(content.keys())):
            f.write(str(verse_id) + '\t' + content[verse_id].strip() + '\n')

def main(input_path, output_weighted_path, output_uniform_path):
    # Read entire file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    comments, content, _ = data.split_pbc_bible_lines(lines=lines, parse_content=True)

    # First pass: collect character frequencies from valid lines only
    char_counts = collections.Counter()
    for text in content.values():
        char_counts.update(text)
    total_chars = sum(char_counts.values())
    chars = list(char_counts.keys())
    weights = list(char_counts.values())
    if total_chars == 0:
        raise ValueError('The input file contained no verses with characters')
    generate_weighted = lambda length: "".join(random.choices(chars, weights=weights, k=length))

    generate_uniform = lambda length: "".join(random.choices(UNIFORM_ALPHABET, k=length))

    # Second pass: generate and write the two output files
    weighted_content = {key: generate_weighted(len(value)) for key, value in content.items()}
    uniform_content = {key: generate_uniform(len(value)) for key, value in content.items()}
    write_output(comments, weighted_content, output_weighted_path)
    write_output(comments, uniform_content, output_uniform_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    input_base_path = os.path.basename(input_file)
    output_file_weighted = os.path.join(output_dir, 'gibberish_wgt_' + input_base_path)
    output_file_uniform = os.path.join(output_dir, 'gibberish_uni_' + input_base_path)

    main(input_file, output_file_weighted, output_file_uniform)
