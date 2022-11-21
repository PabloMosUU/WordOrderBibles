import sys

def shortest_unseen_substrings(text) -> list:
    # Make the stupidest possible implementation
    # "text" is actually any iterable, so a string (iterable of chars) or a list of tokens
    l_i_s = []
    for i, ch in enumerate(text):
        longest_match = 0
        for l in range(1, len(text) - i + 1):
            found = False
            for j in range(0, i - l + 1):
                if text[i:i+l] == text[j:j+l]:
                    found = True
                    break
            if not found:
                break
            longest_match = l
        l_i_s.append(longest_match + 1)
    return l_i_s

def read_and_compute(filename: str) -> None:
    with open(filename, 'r') as f:
        text = f.read()
    tokens = text.split(' ')
    shortest_unseen_substring_lengths = shortest_unseen_substrings(tokens)
    with open(filename.split('/')[-1].split('.')[0] + '_li.txt', 'w') as f:
        for i, l in enumerate(shortest_unseen_substring_lengths):
            ch = tokens[i]
            f.write(ch + '\t' + str(l) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'ERROR: usage: {sys.argv[0]} <filename>')
    filename = sys.argv[1]
    #filename = "randomWikipediaSample.txt"
    read_and_compute(filename)
