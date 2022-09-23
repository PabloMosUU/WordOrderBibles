import pandas as pd

def load_embeddings(file: str) -> dict:
    embedding_df = pd.read_csv(file, sep=" ", quoting=3,
                               header=None, index_col=0, na_filter=False)
    return {key: val.values for key, val in embedding_df.T.items()}

if __name__ == '__main__':
    embed_dim = 300
    glove_dir = '/home/pablo/Documents/tools/Glove'
    glove_embedding = load_embeddings(f'{glove_dir}/glove.6B.{embed_dim}d.txt')
    print('finished')
