print("Beginning imports...")

# conditional_train_minimal.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import os # For creating output dir

# Assume these are in the current path or installed
from evodiff.utils import Tokenizer
from evodiff.losses import D3PMLVBLoss, D3PMCELoss
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK, GAP, START, STOP, SEP # Or your specific alphabet
from sequence_models.constants import MASK, MSA_PAD, MSA_ALPHABET, MSA_AAS, GAP, START, STOP, SEP


from collator import ConditionalD3PMCollator
from data import ConditionalProteinDataset
from model import ConditionalByteNetLMTime

# --- 1. User loads data here ---
# ############################################################################
# ## V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V  ##
# ##                  REPLACE WITH YOUR ACTUAL DATA LOADING                 ##

# Example with dummy data (replace with loading from your files):
# sequences = ["MKTV", "VQLLE", "ALTREM", "AELE", "QALR", "VARELQ", "ALPE", "ALRPQE", "ARPQE", "RRRP"] # List of strings
# labels = [1, 2, 1, 3, 2, 1, 1, 3, 2, 1] # List of ints > 0

data = Path('data')

df = pd.read_csv(data / 'train_ec_311.csv')

# Ensure sequences and labels lists are populated here
sequences = df['Sequence'].tolist()
labels = df['label'].astype(int).tolist()

if not sequences or not labels:
     raise ValueError("Please load your sequences and labels lists in the script!")
if len(sequences) != len(labels):
    raise ValueError("Sequences and labels must have the same length!")
if not all(l > 0 for l in labels):
    raise ValueError("Labels must be integers greater than 0!")

print(f"Loaded {len(sequences)} sequences and labels.")
# ## A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A  ##
# ############################################################################

print(PROTEIN_ALPHABET)


# --- 2. Minimal Configuration ---
config = {
    'd_embed': 64,          # Embedding dimension
    'd_model': 128,         # Model dimension
    'n_layers': 4,          # Number of layers
    'kernel_size': 5,       # Kernel size for ByteNet
    'r': 4,                 # Dilation factor r
    'lr': 1e-4,             # Learning rate
    'batch_size': 4,        # Keep small for testing
    'diffusion_timesteps': 50, # Number of noise steps
    'n_classes': max(labels) if labels else 1, # Num actual classes (highest label value)
    'class_dropout_prob': 0.1, # Dropout prob for CFG
    'embedding_scale': 1.0,  # Scale for class embedding addition
    'reweighting_term': 0.01,# Lambda for CE loss term
    'epochs': 2,            # Number of epochs for quick check
    'max_seq_len': 128,     # Max sequence length for truncation
    'log_freq': 5,          # Print loss every N steps
    'output_dir': './minimal_output' # Directory for minimal output
}
print("Configuration loaded.")
os.makedirs(config['output_dir'], exist_ok=True)

# --- 3. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# Use PROTEIN_ALPHABET or your custom alphabet
tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, pad=PAD, all_aas=MSA_AAS, sequences=True)

n_tokens = 28
Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=config['diffusion_timesteps'])
print("Tokenizer and Diffusion matrices initialized.")

pad_id_val = tokenizer.pad_id


# --- 4. Dataset, Collator, DataLoader ---
dataset = ConditionalProteinDataset(
    sequences,
    labels,
    tokenizer=tokenizer,
    max_seq_len=config['max_seq_len']
)
collater = ConditionalD3PMCollator(
    tokenizer=tokenizer,
    num_timesteps=config['diffusion_timesteps'],
    Q=Q_t,
    Q_bar=Q_prod
)
# Use a simple DataLoader
train_loader = DataLoader(
    dataset=dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collater,
    num_workers=0 # Set to 0 for simplicity in testing
)
print("Dataset and DataLoader ready.")

# --- 5. Model, Optimizer, Loss ---
model = ConditionalByteNetLMTime(
    n_tokens=n_tokens,
    d_embedding=config['d_embed'],
    d_model=config['d_model'],
    n_layers=config['n_layers'],
    kernel_size=config['kernel_size'],
    r=config['r'],
    n_classes=config['n_classes'],
    class_dropout_prob=config['class_dropout_prob'],
    embedding_scale=config['embedding_scale'],
    timesteps=config['diffusion_timesteps']
).to(device)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

optimizer = Adam(model.parameters(), lr=config['lr'])
# Using D3PM loss functions as in the original script
loss_func_lvb = D3PMLVBLoss(tmax=config['diffusion_timesteps'], tokenizer=tokenizer).to(device)
loss_func_ce = D3PMCELoss(tokenizer=tokenizer).to(device)
_lambda = config['reweighting_term']
# Keep GradScaler for potential GPU testing / mixed precision
scaler = GradScaler(enabled=(device.type == 'cuda'))
print("Optimizer, Loss, Scaler ready.")

# --- 6. Minimal Training Loop ---
model.train()
total_steps = 0
print("Starting minimal training loop...")
for epoch in range(config['epochs']):
    print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
    for i, batch in enumerate(train_loader):
        total_steps += 1
        # Data unpacking (matches ConditionalD3PMCollater output)
        # Ignores Q, Q_bar, q_x returned by collator as they aren't needed directly here
        src, src_onehot, timestep, tgt, tgt_onehot, _, _, _, class_labels = batch

        # Move data to device
        src, src_onehot, timestep, tgt, tgt_onehot, class_labels = \
            src.to(device), src_onehot.to(device), timestep.to(device), \
            tgt.to(device), tgt_onehot.to(device), class_labels.to(device)

        # Create mask needed for loss/model (masking out padding)
        # Assuming pad_id is accessible from tokenizer
        input_mask = (src != tokenizer.pad_id) # Shape: [batch_size, seq_len]

        optimizer.zero_grad()

        # Use autocast for mixed precision (efficient on GPU)
        with autocast(enabled=(device.type == 'cuda')):
            if any(step < 0 or step >= 50 for step in timestep):
                raise ValueError(f'oh no! timestep so big: {timestep}')

            # Note the inputs to the model: x, y (timestep), class_labels, input_mask
            outputs = model(src, timestep, class_labels=class_labels, input_mask=input_mask.unsqueeze(-1))

            # Calculate loss
            # Ensure input_mask is correctly shaped if needed by loss fns
            # D3PM losses might expect mask without channel dim - check loss fn implementation if errors occur
            lvb_loss = loss_func_lvb(src_onehot, None, outputs, tgt, tgt_onehot, input_mask, timestep, Q_t.to(device), Q_prod.to(device)) # Pass None for q if not used
            ce_loss = loss_func_ce(outputs, tgt, input_mask) # Assumes this handles averaging correctly

            # Combine losses - D3PM often minimizes LVB + lambda * CE (per sequence/batch avg?)
            # Using simple sum here, assuming losses are already averaged appropriately internally
            loss = lvb_loss + (_lambda * ce_loss)
            nll_loss = ce_loss # Keep track of NLL/CE part

        # Backward pass
        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
             print(f"Warning: Loss does not require grad at step {total_steps}. Skipping backward/step.")

        # Minimal Logging
        if total_steps % config['log_freq'] == 0:
            print(f"  Step {total_steps} | Loss: {loss.item():.4f} | NLL (CE): {nll_loss.item():.4f}")

        # Optional: Stop after a small number of steps for a quick test
        # if total_steps >= 20:
        #    break
    # if total_steps >= 20: # Break outer loop too
    #    break

print("\nMinimal training loop finished.")