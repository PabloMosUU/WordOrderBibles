import collections

import compression_entropy
import unittest

import os
import string

# TODO: replace by mocking
MISMATCHER_PATH = '../../KoplenigEtAl/shortestmismatcher.jar'

class TestCompressionEntropy(unittest.TestCase):
    def test_create_random_word(self):
        word = 'something'
        new_word = compression_entropy.create_random_word(len(word),
                                                          string.ascii_letters,
                                                          [1] * len(string.ascii_letters))
        self.assertNotEqual(word, new_word)
        self.assertEqual(len(word), len(new_word))
        self.assertTrue(all([ch.strip() != '' for ch in new_word]))

    def test_to_file(self):
        text = 'some text'
        orig_filename = 'temp.txt'
        expected = 'temp_pablo.txt'
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
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename,
                                                              True,
                                                              MISMATCHER_PATH)
        self.assertEqual([1, 1, 1, 1, 3, 2, 2, 1], mismatch_lengths)
        os.remove(preprocessed_filename)

    def test_run_mismatcher_with_newline(self):
        preprocessed_filename = 'temp.txt'
        with open(preprocessed_filename, 'w') as f:
            f.write('manzanas\nno')
        mismatch_lengths = compression_entropy.run_mismatcher(preprocessed_filename,
                                                              True,
                                                              MISMATCHER_PATH)
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

    def test_select_samples(self):
        sample_sequences = {1: [[0, 2, 81, 9], [4, 7, 3]], 2: [[3, 1, 9]], 3: [[4, 3, 1, 958, 7], [0, 3, 9, 3, 0]]}
        sample_sequences = {k: [[str(w) for w in seq] for seq in v] for k, v in sample_sequences.items()}
        chosen_sample_ids = [1, 3]
        truncate_samples = True
        selected_sample_sequences = compression_entropy.select_samples(sample_sequences,
                                                                       chosen_sample_ids,
                                                                       truncate_samples)
        self.assertEqual({1, 3}, set(selected_sample_sequences.keys()))
        self.assertEqual({1: [['0', '2', '81', '9'], ['4', '7', '3']], 3: [['4', '3', '1', '958', '7'], ['0']]},
                         selected_sample_sequences)

    def test_select_samples_empty(self):
        sample_sequences = collections.defaultdict(list)
        sample_sequences[0] = [str(el) for el in [0, 1, 2]]
        chosen_sample_ids = [0, 1]
        truncate_samples = False
        selected_samples = compression_entropy.select_samples(sample_sequences, chosen_sample_ids, truncate_samples)
        self.assertEqual({0: [str(el) for el in [0, 1, 2]]}, selected_samples)

    def test_get_char_distribution(self):
        text = 'diccionario'
        distrib = compression_entropy.get_char_distribution(text)
        expected = {'d': 1, 'i': 3, 'c': 2, 'o': 2, 'n': 1, 'a': 1, 'r': 1}
        self.assertEqual(expected, distrib)


if __name__ == "__main__":
    unittest.main()
