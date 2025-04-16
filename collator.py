import torch
import numpy as np

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

# this is sourced directly from evodiff. it is here because we need to make a small change and use
# tokenizeMSA instead of tokenize to avoid headaches.
class D3PMCollater(object):
    """
    D3PM Collater for generating batch data according to markov process according to Austin et al.
    inputs:
        sequences : list of sequences
        tokenizer: Tokenizer()
        masking scheme: 'BLOSUM' uses blosum matrix, 'RANDOM' uses uniform transition matrix
        num_timesteps: number of diffusion timesteps

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
        Q : markov matrix
        q_x : forward transition probabilities
    """
    def __init__(self, tokenizer=Tokenizer(), num_timesteps=100, Q=None, Q_bar=None):
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps # Only needed for markov trans, doesnt depend on seq len
        self.K = self.tokenizer.K
        self.Q = Q
        self.Q_bar =Q_bar

    def __call__(self, sequences):
        # Pre pad one-hot arrays
        pad_one_hot = torch.zeros((self.K))

        tokenized = [torch.tensor(self.tokenizer.tokenizeMSA(s)) for s in sequences]

        max_len = max(len(t) for t in tokenized)

        one_hot = pad_one_hot.repeat((len(tokenized), max_len, 1))
        ## This is to deal with an empty sequence ##
        del_index = None
        for i,t in enumerate(tokenized):
            if len(t) == 0:
               del_index = i
            else:
                one_hot[i, :len(t), :] = self.tokenizer.one_hot(t)
        if del_index is not None:
           tokenized.pop(del_index)
           one_hot = torch.cat((one_hot[:del_index], one_hot[del_index + 1:]))
        one_hot = one_hot.to(torch.double)
        src=[]
        timesteps = []
        q_x = pad_one_hot.repeat((len(tokenized), max_len, 1))
        src_one_hot = pad_one_hot.repeat((len(tokenized), max_len, 1))
        for i,t in enumerate(tokenized): # enumerate over batch
            D = len(t)  # sequence length
            x = one_hot[i, :D, :] #self.tokenizer.one_hot(t)
            t = np.random.randint(1, self.num_timesteps) # randomly sample timestep
            # Append timestep
            timesteps.append(t)
            # Calculate forward at time t and t-1
            x_t, q_x_t = sample_transition_matrix(x, self.Q_bar[t]) # x = tgt, x_t = src, Q_bar[t] is cum prod @ time t
            src.append(x_t)
            src_one_hot[i, :D, :] = self.tokenizer.one_hot(x_t)
            q_x[i, :D, :] = q_x_t
        src = _pad(src, self.tokenizer.pad_id)
        tokenized = _pad(tokenized, self.tokenizer.pad_id)
        return (src.to(torch.long), src_one_hot.to(torch.double), torch.tensor(timesteps), tokenized.to(torch.long),
                one_hot.to(torch.double), self.Q, self.Q_bar, q_x.to(torch.double)) #, q_x_minus1.to(torch.double))
