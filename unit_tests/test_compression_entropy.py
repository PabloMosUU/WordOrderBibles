import compression_entropy
import unittest

import os
import string

class TestCompressionEntropy(unittest.TestCase):
    def test_mask_word_structure(self):
        verse_tokens = [['I', 'am', 'nice', 'and', 'clean', 'and', 'that', 'matters', '.']]
        shuffled = compression_entropy.mask_word_structure(verse_tokens, string.ascii_letters)
        self.assertEqual(1, len(shuffled))
        self.assertEqual(len(verse_tokens[0]), len(shuffled[0]))
        self.assertTrue(all([verse_tokens[0][i] != shuffled[0][i] for i in range(len(shuffled[0]))]))
        self.assertTrue(all([len(verse_tokens[0][i]) == len(shuffled[0][i]) \
                             for i in range(len(shuffled[0]))]))
        self.assertEqual(shuffled[0][3], shuffled[0][5])

    def test_mask_word_structure_across_verses(self):
        verse_tokens = [['This', 'is', 'it'], ['It', 'is', 'good']]
        shuffled = compression_entropy.mask_word_structure(verse_tokens, string.ascii_letters)
        self.assertEqual(2, len(shuffled))
        self.assertEqual(shuffled[0][1], shuffled[1][1])


    def test_create_random_word(self):
        word = 'something'
        new_word = compression_entropy.create_random_word(word, string.ascii_letters)
        self.assertNotEqual(word, new_word)
        self.assertEqual(len(word), len(new_word))
        self.assertTrue(all([ch.strip() != '' for ch in new_word]))


    def test_join_verses(self):
        verse_tokens = [['I', 'hate', 'this'], ['I', 'love', 'this']]
        text = compression_entropy.join_verses(verse_tokens)
        self.assertEqual('I hate this I love this', text)


    def test_to_file(self):
        text = 'some text'
        orig_filename = '/home/pablo/temp.txt'
        expected = '/home/pablo/temp_pablo.txt'
        appendix = 'pablo'
        new_filename = compression_entropy.to_file(text, orig_filename, appendix)
        self.assertEqual(expected, new_filename)
        with open(new_filename, 'r') as f:
            self.assertEqual(text, f.read())
        os.remove(expected)


    def test_run_mismatcher(self):
        preprocessed_filename = 'temp.txt'
        with open(preprocessed_filename, 'w') as f:
            f.write('manzanas')
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename, True)
        self.assertEqual([1, 1, 1, 1, 3, 2, 2, 1], mismatch_lengths)
        os.remove(preprocessed_filename)

    def test_run_mismatcher_with_newline(self):
        preprocessed_filename = 'temp.txt'
        with open(preprocessed_filename, 'w') as f:
            f.write('manzanas\nno')
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename, True)
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
