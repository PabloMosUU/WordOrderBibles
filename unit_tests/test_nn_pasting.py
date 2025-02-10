import nn_pasting as wp
import unittest
from nn_pasting import TaggedWord


class TestWordPasting(unittest.TestCase):

    def test_replace_top_nnc(self):
        verses = [[TaggedWord("Description", "NOUN"), TaggedWord("of", "ADP"), TaggedWord("your", "PRON"),
                   TaggedWord("new", "ADJ"), TaggedWord("TWiki", "PROPN"), TaggedWord("site", "NOUN"),
                   TaggedWord(".", "PUNCT")],
                  [TaggedWord("To", "PART"), TaggedWord("learn", "VERB"), TaggedWord("more", "ADJ"),
                   TaggedWord("about", "ADP"), TaggedWord("TWiki", "PROPN"), TaggedWord(",", "PUNCT"),
                   TaggedWord("visit", "VERB"), TaggedWord("the", "DET"), TaggedWord("new", "ADJ"),
                   TaggedWord("TWiki", "PROPN"), TaggedWord("web", "NOUN"), TaggedWord(".", "PUNCT"),
                   TaggedWord("This", "PRON"), TaggedWord("is", "AUX"), TaggedWord("not", "PART"),
                   TaggedWord("a", "DET"), TaggedWord("TWiki", "PROPN"), TaggedWord("site", "NOUN"),
                   TaggedWord(".", "PUNCT")]]
        expected = [[TaggedWord("Description", "NOUN"), TaggedWord("of", "ADP"), TaggedWord("your", "PRON"),
                     TaggedWord("new", "ADJ"), TaggedWord("TWiki site", "NOUN"),
                     TaggedWord(".", "PUNCT")],
                    [TaggedWord("To", "PART"), TaggedWord("learn", "VERB"), TaggedWord("more", "ADJ"),
                     TaggedWord("about", "ADP"), TaggedWord("TWiki", "PROPN"),
                     TaggedWord(",", "PUNCT"), TaggedWord("visit", "VERB"), TaggedWord("the", "DET"),
                     TaggedWord("new", "ADJ"), TaggedWord("TWiki", "PROPN"),
                     TaggedWord("web", "NOUN"), TaggedWord(".", "PUNCT"), TaggedWord("This", "PRON"),
                     TaggedWord("is", "AUX"), TaggedWord("not", "PART"), TaggedWord("a", "DET"),
                     TaggedWord("TWiki site", "NOUN"), TaggedWord(".", "PUNCT")]]
        replaced = wp.replace_top_nnc(verses, ['NOUN', 'PROPN'])
        self.assertEqual(expected, replaced)

    def test_replace_top_nnc_all_merged(self):
        verses = [[TaggedWord("Description", "NOUN"), TaggedWord("of", "ADP"), TaggedWord("your", "PRON"),
                   TaggedWord("new", "ADJ"), TaggedWord("TWikisite", "NOUN"), TaggedWord(".", "PUNCT")],
                  [TaggedWord("To", "PART"), TaggedWord("learn", "VERB"), TaggedWord("more", "ADJ"),
                   TaggedWord("about", "ADP"), TaggedWord("TWiki", "PROPN"), TaggedWord(",", "PUNCT"),
                   TaggedWord("visit", "VERB"), TaggedWord("the", "DET"), TaggedWord("new", "ADJ"),
                   TaggedWord("TWikiweb", "NOUN"), TaggedWord(".", "PUNCT"), TaggedWord("This", "PRON"),
                   TaggedWord("is", "AUX"), TaggedWord("not", "PART"), TaggedWord("a", "DET"),
                   TaggedWord("TWikisite", "NOUN"), TaggedWord(".", "PUNCT")]]
        expected = []
        replaced = wp.replace_top_nnc(verses, ['NOUN', 'PROPN'])
        self.assertEqual(expected, replaced)

    def test_merge_positions(self):
        verses = [[TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the", "DET"),
                   TaggedWord("nightlife", "NOUN"), TaggedWord("and", "CCONJ"), TaggedWord("I", "PRON"),
                   TaggedWord("do", "AUX"), TaggedWord("not", "PART"), TaggedWord("make", "VERB"),
                   TaggedWord("a", "DET"), TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"),
                   TaggedWord("about", "ADP"), TaggedWord("it", "PRON")],
                  [TaggedWord("Belgium", "PROPN"), TaggedWord("plays", "VERB"), TaggedWord("ugly", "ADV")],
                  [TaggedWord("No", "DET"), TaggedWord("hubo", "ADJ"), TaggedWord("otro", "PROPN"),
                   TaggedWord("como", "PROPN"), TaggedWord("Forlan", "PROPN")]]
        positions = [(0, 4), (0, 7), (2, 1)]
        merged = wp.merge_positions(verses, positions)
        expected = [[TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the", "DET"),
                     TaggedWord("nightlife", "NOUN"), TaggedWord("and I", "PRON"),
                     TaggedWord("do", "AUX"), TaggedWord("not make", "VERB"),
                     TaggedWord("a", "DET"), TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"),
                     TaggedWord("about", "ADP"), TaggedWord("it", "PRON")],
                    [TaggedWord("Belgium", "PROPN"), TaggedWord("plays", "VERB"), TaggedWord("ugly", "ADV")],
                    [TaggedWord("No", "DET"), TaggedWord("hubo otro", "PROPN"),
                     TaggedWord("como", "PROPN"), TaggedWord("Forlan", "PROPN")]]
        self.assertEqual(expected, merged)

    def test_join_words(self):
        tokens = [TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the", "DET"),
                  TaggedWord("nightlife", "NOUN"), TaggedWord("and", "CCONJ"), TaggedWord("I", "PRON"),
                  TaggedWord("do", "AUX"), TaggedWord("not", "PART"), TaggedWord("make", "VERB"),
                  TaggedWord("a", "DET"), TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"),
                  TaggedWord("about", "ADP"), TaggedWord("it", "PRON")]
        expected = [TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the nightlife", "NOUN"),
                    TaggedWord("and", "CCONJ"), TaggedWord("I do", "AUX"),
                    TaggedWord("not", "PART"), TaggedWord("make", "VERB"), TaggedWord("a", "DET"),
                    TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"), TaggedWord("about", "ADP"),
                    TaggedWord("it", "PRON")]
        joined = wp.join_words(tokens, [5, 2])
        self.assertEqual(expected, joined)

    def test_join_words_copy(self):
        # Check that a copy was made even of the untouched verses
        tokens = [TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the", "DET"),
                  TaggedWord("nightlife", "NOUN"), TaggedWord("and", "CCONJ"), TaggedWord("I", "PRON"),
                  TaggedWord("do", "AUX"), TaggedWord("not", "PART"), TaggedWord("make", "VERB"),
                  TaggedWord("a", "DET"), TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"),
                  TaggedWord("about", "ADP"), TaggedWord("it", "PRON")]
        expected = [TaggedWord("I", "PRON"), TaggedWord("love", "VERB"), TaggedWord("the", "DET"),
                    TaggedWord("nightlife", "NOUN"), TaggedWord("and", "CCONJ"), TaggedWord("I", "PRON"),
                    TaggedWord("do", "AUX"), TaggedWord("not", "PART"), TaggedWord("make", "VERB"),
                    TaggedWord("a", "DET"), TaggedWord("big", "ADJ"), TaggedWord("fuss", "NOUN"),
                    TaggedWord("about", "ADP"), TaggedWord("it", "PRON")]
        joined = wp.join_words(tokens, [])
        self.assertEqual(expected, joined)

    def test_create_word_pasted_sets(self):
        id_verses = {0: [[TaggedWord('I', 'NOUN'), TaggedWord('am', 'NOUN'), TaggedWord('here', 'NOUN'),
                          TaggedWord('I', 'NOUN'), TaggedWord('am', 'VERB')]],
                     1: [[TaggedWord('I am here', 'VERB')]]}
        steps_to_save = {0, 1}
        expected = {0: {0: [['I', 'am', 'here', 'I', 'am']], 1: [['I am', 'here', 'I', 'am']]},
                    1: {0: [['I am here']]}}
        word_pasted_sets = wp.create_word_pasted_sets(id_verses, steps_to_save, ['NOUN', 'PROPN'])
        self.assertEqual(expected, word_pasted_sets)


if __name__ == "__main__":
    unittest.main()
