from collections import defaultdict, Counter

class BPE:
    def __init__(self, text: str):
        self.text = text
        tokens = text.split()
        self.token_pair_locations = defaultdict(list)
        self.token_pair_counter = Counter()
        self.unique_tokens = {tokens[0]}
        for i in range(0, len(tokens) - 1):
            token_pair = (tokens[i], tokens[i+1])
            self.token_pair_locations[token_pair].append(i)
            self.token_pair_counter.update(token_pair)
            self.unique_tokens.add(tokens[i+1])

    def step(self):
        raise NotImplementedError()

    def get_n_tokens(self):
        raise NotImplementedError()
