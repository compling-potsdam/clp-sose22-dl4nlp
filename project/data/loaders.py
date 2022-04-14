import os

import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def get_tokenizer():
    return torchtext.data.utils.get_tokenizer("basic_english")


def load_imdb_vocab(data_dir, filename="imdb"):
    vocab_path = os.path.join(data_dir, "IMDB/aclImdb", filename + ".vocab")
    vocab = {"itos": dict(), "stoi": dict()}
    with open(vocab_path, "r") as f:
        for idx, word in enumerate(f.readlines()):
            vocab["itos"][idx] = word.strip()  # remove newlines
            vocab["stoi"][word.strip()] = idx
    vocab_size = len(vocab["itos"])  # append special tokens at the end
    for idx, special_token in enumerate(["<s>", "<e>", "<p>"]):
        vocab["itos"][vocab_size + idx] = special_token
        vocab["stoi"][special_token] = vocab_size + idx
    return vocab


def load_imbd_samples(data_dir, split, category, dry_run=False):
    data_path = os.path.join(data_dir, "IMDB/aclImdb", split, category)
    samples = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r") as f:
            line = f.readline()
            samples.append((line, category))
        if dry_run and len(samples) >= 100:
            break
    return samples


def download_imbdb_data(data_dir):
    torchtext.datasets.IMDB(root=data_dir, split="train")
    torchtext.datasets.IMDB(root=data_dir, split="test")


class ImdbDataset(torch.utils.data.Dataset):
    """
        DataPipe that yields tuple of label (1 to 2) and text containing the movie review
    """

    def __init__(self, vocab, split: str, data_dir: str, dry_run: bool = False):
        self.data_dir = data_dir
        self.vocab = vocab
        self.tokenizer = get_tokenizer()
        self.start_token = vocab["stoi"]["<s>"]
        self.end_token = vocab["stoi"]["<e>"]
        self.pad_token = vocab["stoi"]["<p>"]
        self.split = split
        self.positives = load_imbd_samples(data_dir, split, "pos", dry_run)
        self.negatives = load_imbd_samples(data_dir, split, "neg", dry_run)
        self.samples = self.positives + self.negatives
        if split == "train":
            self.unsupported = load_imbd_samples(data_dir, split, "unsup", dry_run)
            self.samples += self.unsupported
        self.ltoi = {"pos": 0, "neg": 1, "unsup": 2}

    def __get_inputs(self, index):
        text, _ = self.samples[index]
        tokens = self.tokenizer(text)
        encoding = []
        for token in tokens:
            # we hoped that our tokenizer matches their vocab, but they removed punctation
            if token in self.vocab["stoi"]:
                encoding.append(self.vocab["stoi"][token])
        encoding = [self.start_token] + encoding + [self.end_token]
        return torch.tensor(encoding)

    def __get_labels(self, index):
        _, label = self.samples[index]
        label_idx = self.ltoi[label]
        return torch.tensor(label_idx)

    def __getitem__(self, index):
        inputs = self.__get_inputs(index)
        labels = self.__get_labels(index)
        return inputs, labels

    def collate(self, data):
        inputs = [d[0] for d in data]  # varying length sequence
        inputs = pad_sequence(sequences=inputs, padding_value=self.pad_token, batch_first=True)
        labels = [d[1] for d in data]
        labels = default_collate(labels)
        return inputs, labels

    def __len__(self):
        return len(self.samples)
