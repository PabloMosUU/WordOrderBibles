import compression_entropy
import unittest

import data
import os


class TestCompressionEntropy(unittest.TestCase):
    def test_mask_word_structure(self):
        verse_tokens = {1: ['I', 'am', 'nice', 'and', 'clean', 'and', 'that', 'matters', '.']}
        tokenized_bible = data.TokenizedBible('eng', 'eng_test.txt', verse_tokens)
        shuffled = compression_entropy.mask_word_structure(tokenized_bible)
        self.assertEqual(1, len(shuffled))
        self.assertTrue(1 in shuffled)
        self.assertEqual(len(verse_tokens[1]), len(shuffled[1]))
        self.assertTrue(all([verse_tokens[1][i] != shuffled[1][i] for i in range(len(shuffled[1]))]))
        self.assertTrue(all([len(verse_tokens[1][i]) == len(shuffled[1][i]) \
                             for i in range(len(shuffled[1]))]))
        self.assertEqual(shuffled[1][3], shuffled[1][5])

    def test_mask_word_structure_across_verses(self):
        verse_tokens = {1: ['This', 'is', 'it'], 2: ['It', 'is', 'good']}
        tokenized_bible = data.TokenizedBible('eng', 'eng_test_2.txt', verse_tokens)
        shuffled = compression_entropy.mask_word_structure(tokenized_bible)
        self.assertEqual(2, len(shuffled))
        self.assertTrue(1 in shuffled and 2 in shuffled)
        self.assertEqual(shuffled[1][1], shuffled[2][1])


    def test_create_random_word(self):
        word = 'something'
        new_word = compression_entropy.create_random_word(word)
        self.assertNotEqual(word, new_word)
        self.assertEqual(len(word), len(new_word))
        self.assertTrue(all([ch.strip() != '' for ch in new_word]))


    def test_join_verses(self):
        verse_tokens = {'3': ['I', 'love', 'this'], '2': ['I', 'hate', 'this']}
        text = compression_entropy.join_verses(verse_tokens)
        self.assertEqual('I hate this I love this', text)


    def test_to_file(self):
        text = 'some text'
        orig_filename = '/path/to/temp.txt'
        appendix = 'pablo'
        new_filename = compression_entropy.to_file(text, orig_filename, appendix)
        self.assertEqual('temp_pablo.txt', new_filename)
        with open(new_filename, 'r') as f:
            self.assertEqual(text, f.read())
        os.remove('temp_pablo.txt')


    def test_run_mismatcher(self):
        preprocessed_filename = 'temp.txt'
        with open(preprocessed_filename, 'w') as f:
            f.write('manzanas')
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename)
        self.assertEqual([1, 1, 1, 1, 3, 2, 2, 1], mismatch_lengths)
        os.remove(preprocessed_filename)

    def test_run_mismatcher_with_newline(self):
        preprocessed_filename = 'temp.txt'
        with open(preprocessed_filename, 'w') as f:
            f.write('manzanas\nno')
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename)
        self.assertEqual([1, 1, 1, 1, 3, 2, 2, 1, 1, 2, 1], mismatch_lengths)
        os.remove(preprocessed_filename)

    def test_mismatcher_lines(self):
        lines = ['a\t1\n', 'b\t2\n']
        self.assertEqual([1, 2], compression_entropy.parse_mismatcher_lines(lines))

    def test_mismatcher_lines_with_newline(self):
        lines = ['a\t1\n', '\n', '\t2\n', 't\t3\n']
        self.assertEqual([1, 2, 3], compression_entropy.parse_mismatcher_lines(lines))

    def test_mismatcher_lines_with_tab(self):
        lines = ['a\t1\n', '\t\t3\n']
        self.assertEqual([1, 3], compression_entropy.parse_mismatcher_lines(lines))

    def test_entropy(self):
        mismatches = [1, 2, 6, 3, 2]
        entropy = compression_entropy.get_entropy(mismatches)
        self.assertAlmostEqual(0.61373, entropy, 4)


if __name__ == "__main__":
    unittest.main()
