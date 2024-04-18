import unittest
import word_splitting
from util import Token


class TestBpe(unittest.TestCase):
    def test_has_completed_merges(self):
        text = "One line\nAnother line's line"
        verses = text.split('\n')
        verse_tokens = [el.split(' ') for el in verses]
        assert any(['\'' in el for el in verses])
        bpe_tokenizer = word_splitting.train_tokenizer(verse_tokens, 100)
        is_merge_complete = word_splitting.has_completed_merges(verse_tokens, bpe_tokenizer)
        self.assertEqual(True, is_merge_complete)

    def test_get_merge_steps(self):
        merge_steps = word_splitting.get_merge_steps('merge_steps.txt')
        self.assertEqual([['h', 'e'], ['t', 'he'], ['a', 'n'], ['an', 'd'], ['o', 'u'], ['i', 'n']], merge_steps)

    def test_split_chars(self):
        seq_token_sub_tokens = word_splitting.split_chars([
            ['God', 'Kaun', 'ak', 'abe', ',', 'Bin', 'Bɨ', 'dɨnɨgab', 'Bin', 'nɨbak', 'abe'],
            ['Bin', 'bɨ', 'mɨnɨm', 'buk', 'nɨbaul', 'mɨdeb', 'udɨn', 'lɨ', 'nɨŋɨl', 'agel']
        ])
        self.assertEqual([[['G', 'o', 'd'], ['K', 'a', 'u', 'n'], ['a', 'k'], ['a', 'b', 'e'], [','], ['B', 'i', 'n'],
                           ['B', 'ɨ'], ['d', 'ɨ', 'n', 'ɨ', 'g', 'a', 'b'], ['B', 'i', 'n'], ['n', 'ɨ', 'b', 'a', 'k'],
                           ['a', 'b', 'e']],
                          [['B', 'i', 'n'], ['b', 'ɨ'], ['m', 'ɨ', 'n', 'ɨ', 'm'], ['b', 'u', 'k'],
                           ['n', 'ɨ', 'b', 'a', 'u', 'l'], ['m', 'ɨ', 'd', 'e', 'b'], ['u', 'd', 'ɨ', 'n'], ['l', 'ɨ'],
                           ['n', 'ɨ', 'ŋ', 'ɨ', 'l'], ['a', 'g', 'e', 'l']]],
                         seq_token_sub_tokens)

    def test_build_merge_history(self):
        seq_tokens = [['Farizhɛɛnbii', 'niɲyahara', 'mpa', 'piye'],
                      ['Yii', 'a']]
        merge_steps = [['ɛ', 'ɛ'], ['a', 'p'], ['n', 'y'], ['ɛɛ', 'n']]
        merge_history = word_splitting.build_merge_history(seq_tokens, merge_steps)
        expected_0 = [[['F', 'a', 'r', 'i', 'z', 'h', 'ɛ', 'ɛ', 'n', 'b', 'i', 'i'],
                       ['n', 'i', 'ɲ', 'y', 'a', 'h', 'a', 'r', 'a'],
                       ['m', 'p', 'a'], ['p', 'i', 'y', 'e']],
                      [['Y', 'i', 'i'], ['a']]]
        expected_1 = [[['F', 'a', 'r', 'i', 'z', 'h', 'ɛɛ', 'n', 'b', 'i', 'i'],
                       ['n', 'i', 'ɲ', 'y', 'a', 'h', 'a', 'r', 'a'],
                       ['m', 'p', 'a'], ['p', 'i', 'y', 'e']],
                      [['Y', 'i', 'i'], ['a']]]
        expected_2 = [[['F', 'a', 'r', 'i', 'z', 'h', 'ɛɛ', 'n', 'b', 'i', 'i'],
                       ['n', 'i', 'ɲ', 'y', 'a', 'h', 'a', 'r', 'a'],
                       ['m', 'p', 'a'], ['p', 'i', 'y', 'e']],
                      [['Y', 'i', 'i'], ['a']]]
        expected_3 = [[['F', 'a', 'r', 'i', 'z', 'h', 'ɛɛ', 'n', 'b', 'i', 'i'],
                       ['n', 'i', 'ɲ', 'y', 'a', 'h', 'a', 'r', 'a'],
                       ['m', 'p', 'a'], ['p', 'i', 'y', 'e']],
                      [['Y', 'i', 'i'], ['a']]]
        expected_4 = [[['F', 'a', 'r', 'i', 'z', 'h', 'ɛɛn', 'b', 'i', 'i'],
                       ['n', 'i', 'ɲ', 'y', 'a', 'h', 'a', 'r', 'a'],
                       ['m', 'p', 'a'], ['p', 'i', 'y', 'e']],
                      [['Y', 'i', 'i'], ['a']]]
        expected = {4: expected_0, 3: expected_1, 2: expected_2, 1: expected_3, 0: expected_4}
        self.assertEqual(expected.keys(), merge_history.keys())
        for i in expected.keys():
            if expected[i] != merge_history[i]:
                assert len(expected[i]) == len(merge_history[i])
                print('Sequence', i)
                for j in range(len(expected[i])):
                    if expected[i][j] != merge_history[i][j]:
                        print('Token', j)
                        print('Expected:', expected[i][j])
                        print('Real:', merge_history[i][j])
                        raise ValueError()

    def test_build_merge_history_long(self):
        seq_tokens = [['Farizhɛɛnbii', 'ná', 'Sadusiibii', 'niɲyahara', 'mpyi', 'na', 'ma', 'si', 'mpa', 'piye'],
                      ['Yii', 'a', 'katiigii', 'pyi', ',', 'lire', 'sí', 'li', 'cyêe', 'na', 'yii', "zòompil'à"]]
        merge_steps = [[el, el] for el in list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')]
        merge_history = word_splitting.build_merge_history(seq_tokens, merge_steps)
        self.assertEqual(16, len(merge_history))
        self.assertEqual([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 48, 49, 50, 52],
                         sorted(list(merge_history.keys())))

    def test_get_merge_step_points(self):
        merge_step_points = word_splitting.get_merge_step_points(531)
        expected = [0, 53, 106, 159, 212, 265, 318, 371, 424, 477, 482, 488, 493, 498, 504, 509, 514, 519, 525, 530, 531]
        self.assertEqual(expected, merge_step_points)

    def test_apply_merge(self):
        seq_token_sub_tokens = [[['M', 'u', 's', 'a'], ['w', 'e', 'n', 'e'], ['o', 'n', 'o', 'l', 'u', 'k']],
                                [['"'], ["'"], ['A', 'p'], ['u', 'n', 'u', 's', 'u', 'k'], ['f', 'u', 'g'], [',']]]
        merge_step = ['s', 'a']
        merged = word_splitting.apply_merge(seq_token_sub_tokens, merge_step)
        expected = [[['M', 'u', 'sa'], ['w', 'e', 'n', 'e'], ['o', 'n', 'o', 'l', 'u', 'k']],
                    [['"'], ["'"], ['A', 'p'], ['u', 'n', 'u', 's', 'u', 'k'], ['f', 'u', 'g'], [',']]]
        self.assertEqual(expected, merged)

    def test_flatten_sequences(self):
        splits_seq_token_sub_tokens = {10: [[['I'], ['a', 'm']],
                                            [['T', 'h', 'e'], ['p', 'u', 'n', 'k']]]}
        flattened = {10: [[Token('I', True),
                           Token('a', True), Token('m', False)],
                          [Token('T', True), Token('h', False),
                          Token('e', False),
                          Token('p', True), Token('u', False),
                          Token('n', False), Token('k', False)]]}
        self.assertEqual(flattened, word_splitting.flatten_sequences(splits_seq_token_sub_tokens))

    def test_join_verses(self):
        verse_tokens = [['I', 'hate', 'this'], ['I', 'love', 'this']]
        verse_tokens = [[Token(el, True) for el in seq] for seq in verse_tokens]
        text = word_splitting.join_verses(verse_tokens, insert_spaces=True)
        self.assertEqual('I hate this I love this', text)

    def test_join_verses_with_sub_tokens(self):
        verse_tokens = [['I', 'hat', 'e', 'this'], ['I', 'love', 'this']]
        verse_tokens = [[Token(el, True) for el in seq] for seq in verse_tokens]
        verse_tokens[0][2] = Token('e', False)
        text = word_splitting.join_verses(verse_tokens, insert_spaces=True)
        self.assertEqual('I hate this I love this', text)

    def test_join_verses_with_sub_tokens_no_space(self):
        verse_tokens = [['I', 'hat', 'e', 'this'], ['I', 'love', 'this']]
        verse_tokens = [[Token(el, True) for el in seq] for seq in verse_tokens]
        verse_tokens[0][2] = Token('e', False)
        text = word_splitting.join_verses(verse_tokens, insert_spaces=False)
        self.assertEqual('IhatethisIlovethis', text)


if __name__ == "__main__":
    unittest.main()
