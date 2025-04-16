import torch
import numpy as np
from torch.utils.data import Dataset

from evodiff.utils import Tokenizer

class ConditionalProteinDataset(Dataset):
    """
    in their original training, sequence-based data was loaded using sequence_models.UnirefDataset
    since the implementation was so simple, we redefine a torch dataset with explicit class labels instead.
    """
    def __init__(self, sequences, labels, tokenizer=None, max_seq_len=512):
        """
        args:
            sequences: list of sequences
            labels: list of labels
            tokenizer: evodiff Tokenizer object
            max_seq_len: maximum sequence length
        """
        # quick checks
        if not len(sequences) == len(labels):
            raise ValueError(f"Sequences ({len(sequences)}) and labels ({len(labels)}) must have the same length!")
        if not all(label > 0 for label in labels):
            raise ValueError(f"Class labels must be > 0 (0 is reserved for null class!)")

        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        return sequence and class label
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        tokenized_seq = self.tokenizer.tokenizeMSA(sequence)

        # truncate if > max_seq_len. truncate based on # of tokens
        if len(tokenized_seq) > self.max_seq_len:
            tokenized_seq = tokenized_seq[:self.max_seq_len]
        
        # convert back to string
        processed_seq = self.tokenizer.untokenize(tokenized_seq)

        return (processed_seq, label)




            