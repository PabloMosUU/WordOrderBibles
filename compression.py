"""
Dumb but straightforward implementation of the computation of the shortest unseen subsequence lengths
"""
import sys


def shortest_unseen_subsequence_lengths(sequence) -> list:
    """
    Dumb but straightforward implementation of the computation of the shortest unseen subsequence lengths
    :param sequence: a sequence (string, list, etc.)
    :return: a list of lengths of shortest subsequences at each element of the sequence
    """
    l_i_s = []
    for i, ch in enumerate(sequence):
        longest_match = 0
        for ll in range(1, len(sequence) - i + 1):
            found = False
            for j in range(0, i - ll + 1):
                if sequence[i:i + ll] == sequence[j:j + ll]:
                    found = True
                    break
            if not found:
                break
            longest_match = ll
        l_i_s.append(longest_match + 1)
    return l_i_s


def read_and_compute(filename: str) -> None:
    with open(filename, 'r') as f:
        text = f.read()
    tokens = text.split(' ')
    shortest_unseen_substring_lengths = shortest_unseen_subsequence_lengths(tokens)
    with open(filename.split('/')[-1].split('.')[0] + '_li.txt', 'w') as f:
        for i, l in enumerate(shortest_unseen_substring_lengths):
            ch = tokens[i]
            f.write(ch + '\t' + str(l) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'ERROR: usage: {sys.argv[0]} <filename>')
    filename = sys.argv[1]
    read_and_compute(filename)
