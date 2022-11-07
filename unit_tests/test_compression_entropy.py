import compression_entropy
import unittest

import data


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


if __name__ == "__main__":
    unittest.main()
