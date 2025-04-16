import torch
import numpy as np

from evodiff.collaters import D3PMCollater

class ConditionalD3PMCollator(D3PMCollater):
    """
    extension of D3PMCollator that includes class labels for conditional generation.
    they made a spelling mistake in 'collater' which is fixed here.
    """
    def __init__(self, tokenizer, num_timesteps=100, Q=None, Q_bar=None):
        """
        args:
            tokenzier: instance of evodiff.utils.Tokenizer
            num_timesteps: divison of steps in forward process
            Q: pre-defined transition probability matrix
            Q_bar: cumulative product of Q at t
        """
        super().__init__(tokenizer=tokenizer, num_timesteps=num_timesteps, Q=Q, Q_bar=Q_bar)

    def __call__(self, batch_data):
        """
        processes batch of data.
        args:
            batch_data: list of tuples (e.g. [(sequence1, label1), (sequence2, label2), ...])
                        labels have to be >= 1 since 0 is reserved for null class

        returns:
            same as Tokenizer, except with class_labels at the end
        """
        # split sequence and their labels
        sequences, class_labels = zip(*batch_data)

        # process sequences through parent collator
        batch_output = super().__call__(sequences)

        # convert class labels to tensor and add batch output
        class_labels_tensor = torch.tensor(class_labels, dtype=torch.long)

        # return the combined batch data
        return batch_output + (class_labels_tensor,)

