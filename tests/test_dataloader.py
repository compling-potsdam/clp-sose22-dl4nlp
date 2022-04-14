import unittest

from torch.utils.data import DataLoader
from torch import nn

from project.data.loaders import download_imbdb_data, ImdbDataset, load_imdb_vocab


class MyTestCase(unittest.TestCase):
    data_dir = "/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/data"

    def test_download_imbdb_data(self):
        download_imbdb_data(MyTestCase.data_dir)

    def test_imbdb_dataset(self):
        vocab = load_imdb_vocab(MyTestCase.data_dir)
        train_data = ImdbDataset(vocab, "train", MyTestCase.data_dir, dry_run=True)
        loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, collate_fn=train_data.collate)

        vocab_size = len(vocab["stoi"])
        word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512, padding_idx=vocab["stoi"]["<p>"])

        for batch in loader:
            x, y = batch[0], batch[1]
            print("X:", x.shape)
            print("Example X:", x[0][:10])

            print("Y:", y.shape)
            print("Example Y:", y[0])

            word_embeddings = word_embedding(x)
            print("E:", word_embeddings.shape)
            break


if __name__ == '__main__':
    unittest.main()
