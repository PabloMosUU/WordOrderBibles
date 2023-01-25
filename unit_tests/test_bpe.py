import unittest
import bpe_pablo

class TestBpe(unittest.TestCase):
    def test_step(self):
        text = 'manzanas'
        bpe_instance = bpe_pablo.BPE(text)
        bpe_instance.step()
        # Check that an is in the list of tokens, but n is not
        self.assertTrue('an' in bpe_instance.unique_tokens)
        self.assertFalse('n' in bpe_instance.unique_tokens)

if __name__ == "__main__":
    unittest.main()
