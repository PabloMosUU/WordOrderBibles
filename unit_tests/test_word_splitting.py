import unittest
import word_splitting


class TestBpe(unittest.TestCase):
    def test_create_word_split_sets(self):
        id_verses = {0: ['boy boys girl girls '.split(' ')]}
        output = word_splitting.create_word_split_sets(id_verses, 10000)
        self.assertEqual(1, len(output))
        self.assertEqual(0, list(output.keys())[0])
        self.assertEqual(1, len(output[0]))
        self.assertEqual(1, list(output[0].keys())[0])
        self.assertEqual([['boy', 'boy', 's', 'girl', 'girl', 's']], output[0][1])

    def test_has_completed_merges(self):
        text = "One line\nAnother line's line"
        verses = text.split('\n')
        verse_tokens = [el.split(' ') for el in verses]
        assert any(['\'' in el for el in verses])
        bpe_tokenizer = word_splitting.train_tokenizer(verse_tokens, 100)
        is_merge_complete = word_splitting.has_completed_merges(verse_tokens, bpe_tokenizer)
        self.assertEqual(True, is_merge_complete)


if __name__ == "__main__":
    unittest.main()
