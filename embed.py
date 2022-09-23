import pandas as pd

import data
import util

KEY_MAP = {'<unk>': data.UNKNOWN_TOKEN, 'n/a': data.PAD_TOKEN,
           '<': data.START_OF_VERSE_TOKEN, '>': data.END_OF_VERSE_TOKEN}

def replace_keys(word_embedding: dict, key_map: dict) -> dict:
    for old_key, new_key in key_map.items():
        word_embedding = util.replace_key(word_embedding, old_key, new_key)
    return word_embedding

def load_embeddings(file: str) -> dict:
    embedding_df = pd.read_csv(file, sep=" ", quoting=3,
                               header=None, index_col=0, na_filter=False)
    embeddings = {key: val.values.tolist() for key, val in embedding_df.T.items()}
    return replace_keys(embeddings, KEY_MAP)

if __name__ == '__main__':
    embed_dim = 300
    glove_dir = '/home/pablo/Documents/tools/Glove'
    glove_embedding = load_embeddings(f'{glove_dir}/glove.6B.{embed_dim}d.txt')
    print('finished')
