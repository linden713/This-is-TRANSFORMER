import os
import torch
import spacy
import urllib.request
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

def download_multi30k():
    """Download Multi30k dataset if not present"""
    # Create data directory
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download files if they don't exist
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = {
        "train.de": "train.de.gz",
        "train.en": "train.en.gz",
        "val.de": "val.de.gz",
        "val.en": "val.en.gz",
        "test.de": "test_2016_flickr.de.gz",
        "test.en": "test_2016_flickr.en.gz"
    }

    for local_name, remote_name in files.items():
        filepath = f'data/{local_name}'
        if not os.path.exists(filepath):
            url = base_url + remote_name
            urllib.request.urlretrieve(url, filepath + '.gz')
            os.system(f'gunzip -f {filepath}.gz')

def load_data(filename):
    """Load data from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def create_dataset():
    """Create dataset from files"""
    # Download data if needed
    download_multi30k()

    # Load data
    train_de = load_data('data/train.de')
    train_en = load_data('data/train.en')
    val_de = load_data('data/val.de')
    val_en = load_data('data/val.en')

    return (train_de, train_en), (val_de, val_en)

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Tokenize
        src_tokens = [tok.text for tok in self.src_tokenizer(src_text)]
        tgt_tokens = [tok.text for tok in self.tgt_tokenizer(tgt_text)]

        # Convert to indices
        src_indices = [self.src_vocab["<s>"]] + [self.src_vocab[token] for token in src_tokens] + [self.src_vocab["</s>"]]
        tgt_indices = [self.tgt_vocab["<s>"]] + [self.tgt_vocab[token] for token in tgt_tokens] + [self.tgt_vocab["</s>"]]

        return {
            'src': torch.tensor(src_indices),
            'tgt': torch.tensor(tgt_indices)
        }

def build_vocab_from_texts(texts, tokenizer, min_freq=2):
    """Build vocabulary from texts"""
    counter = {}
    for text in texts:
        for token in [tok.text for tok in tokenizer(text)]:
            counter[token] = counter.get(token, 0) + 1

    # Create vocabulary
    vocab_dict = {"<s>": 0, "</s>": 1, "<blank>": 2, "<unk>": 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab_dict[word] = idx
            idx += 1

    vocab = defaultdict(lambda: vocab_dict["<unk>"])
    vocab.update(vocab_dict)
    return vocab

def create_dataloaders(batch_size=32):
    # Load tokenizers
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    # Get data
    (train_de, train_en), (val_de, val_en) = create_dataset()

    # Build vocabularies
    vocab_src = build_vocab_from_texts(train_de, spacy_de)
    vocab_tgt = build_vocab_from_texts(train_en, spacy_en)

    # Create datasets
    train_dataset = TranslationDataset(
        train_de, train_en,
        vocab_src, vocab_tgt,
        spacy_de, spacy_en
    )

    val_dataset = TranslationDataset(
        val_de, val_en,
        vocab_src, vocab_tgt,
        spacy_de, spacy_en
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    return train_dataloader, val_dataloader, vocab_src, vocab_tgt

def collate_batch(batch):
    src_tensors = [item['src'] for item in batch]
    tgt_tensors = [item['tgt'] for item in batch]

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=2)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=2)

    return {
        'src': src_padded,
        'tgt': tgt_padded
    }