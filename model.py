# model with conditioning
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from evodiff.model import ByteNetLMTime, PositionalEncoding1D

class ConditionalByteNetLMTime(ByteNetLMTime):
    """
    Extends original ByteNetLMTime model with classifier-free diffusion guidance capabilities.
    Includes class conditioning and conditional dropout, where:
     - Class 0 is reserved for unconditional generation.
     - Class 1+ represent different desired classes.
    """
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                n_classes, class_dropout_prob=0.1, embedding_scale=1.0, 
                rank=None, n_frozen_embs=None, padding_idx=None, causal=False,
                dropout=0.0, final_ln=False, slim=True, activation='relu',
                tie_weights=False, down_embed=True, timesteps=None):
        super().__init__(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                        rank=rank, n_frozen_embs=n_frozen_embs, padding_idx=padding_idx,
                        causal=causal, dropout=dropout, final_ln=final_ln, slim=slim,
                        activation=activation, tie_weights=tie_weights, down_embed=down_embed,
                        timesteps=timesteps)

        self.timesteps = timesteps

        # add class embedding
        self.n_classes = n_classes
        self.class_dropout_prob = class_dropout_prob
        self.embedding_scale = embedding_scale
        self.class_embedding = nn.Embedding(n_classes + 1, d_model)

    def forward(self, x, y, class_labels=None, input_mask=None):
        """
        Forwadr pass including conditional information.

        Args:
            x: Input sequence tokens [batch_size, seq_len]
            y: Diffusion timestep [batch_size]
            class_labels: Class labels for conditioning [batch_size]
            input_mask: mask for padded regions [batch_size, seq_len, 1]
        Returns:
            predictions
        """
        # get base embeddings according to normal d3pm process
        e = self.embedder._embed(x, y, timesteps=self.timesteps)

        # apply class conditioning if provided
        if class_labels is not None:
            # if we are training, randomly flip class to 0 label
            if self.training:
                dropout_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.class_dropout_prob

                # apply mask and preserve original labels
                conditioning_labels = class_labels.clone()
                conditioning_labels[dropout_mask] = 0

                # get class embeddings
                class_emb = self.class_embedding(conditioning_labels)
            else:
                # during inference, we just use the class labels provided
                class_emb = self.class_embedding(class_labels)
        
            # add class embedding to sequence embedding
            e = e + self.embedding_scale * repeat(class_emb, 'b d -> b s d', s=e.shape[1]) # [batch_size, d_embed] -> [batch_size, seq_len, d_embed]

        # continue with normal forward pass
        e = self.embedder._convolve(e, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)

    def sample(self, x, y, class_labels, guidance_scale=3.0, input_mask=None):
        """
        sample for classifier-free guided sampling.

        args:
            x: input sequence tokens [batch_size, seq_len]
            y: timestep tokens [batch_size]
            class_labels: class labels for conditional generation [batch_size]
            input_mask: mask for padded regions [batch_size, seq_len, 1]
        returns:
            guided model predictions
        """
        # create null class labels (for unconditioned generation part)
        null_labels = torch.zeros_like(class_labels)

        # predict
        with torch.no_grad():
            uncond_pred = self(x, y, class_labels=null_labels, input_mask=input_mask)

            # get conditional prediction
            cond_pred = self(x, y, class_labels=class_labels, input_mask=input_mask)

            # apply guidance formula
            guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        return guided_pred

if __name__ == '__main__':
    vocab_size = 25 # e.g., 20 AAs + pad + mask + etc. + class token?
    d_emb = 32      # Initial embedding dim
    d_model = 64    # Model dim (must match class embedding dim)
    n_layers = 2
    k_size = 3
    r_rate = 4
    num_classes = 3 # Actual classes (0 is null)
    max_t = 50      # Max timesteps

    # 2. Instantiate Model
    try:
        model = ConditionalByteNetLMTime(
            n_tokens=vocab_size, d_embedding=d_emb, d_model=d_model,
            n_layers=n_layers, kernel_size=k_size, r=r_rate,
            n_classes=num_classes, class_dropout_prob=0.1, timesteps=max_t
        )
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"ERROR during model instantiation: {e}")
        exit()

    # 3. Dummy Data
    batch_size = 4
    seq_len = 16
    x_dummy = torch.randint(0, vocab_size, (batch_size, seq_len)) # token ids
    t_dummy = torch.randint(0, max_t, (batch_size,))              # timesteps
    # class labels >= 1 for conditional example
    labels_dummy = torch.randint(1, num_classes + 1, (batch_size,))

    # 4. Test Forward Pass (Training Mode)
    try:
        model.train()
        output_train = model(x_dummy, t_dummy, labels_dummy)
        print(f"Train Forward Output Shape: {output_train.shape}") # Expected: [batch_size, seq_len, vocab_size]
        assert output_train.shape == (batch_size, seq_len, vocab_size)
    except Exception as e:
        print(f"ERROR during training forward pass: {e}")

    # 5. Test CFG Sampling (Eval Mode)
    try:
        model.eval()
        output_cfg = model.sample(x_dummy, t_dummy, labels_dummy, guidance_scale=2.0)
        print(f"CFG Sample Output Shape:  {output_cfg.shape}")   # Expected: [batch_size, seq_len, vocab_size]
        assert output_cfg.shape == (batch_size, seq_len, vocab_size)
    except Exception as e:
        print(f"ERROR during CFG sampling: {e}")

    print("\nSimple sanity check finished.")