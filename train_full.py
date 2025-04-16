print('first round imports')
# conditional_train_single_gpu.py
import argparse
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pathlib
import torch
# Removed: multiprocessing, distributed, DDP, DistributedSampler
print('second round imports')

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, Dataset # Removed DistributedSampler
from torch.cuda.amp import GradScaler, autocast

print('third round imports')

# Assume these are in the current path or installed
from evodiff.utils import Tokenizer
from evodiff.losses import D3PMLVBLoss, D3PMCELoss
# Use your specific sequence constants
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK, GAP, START, STOP, SEP, MSA_AAS


print('fourth round imports')

# Import your custom components
from collator import ConditionalD3PMCollator # Assuming collator.py exists
from data import ConditionalProteinDataset     # Assuming data.py exists
from model import ConditionalByteNetLMTime    # Assuming model.py exists

print('done')

# Helper function for LR scheduling (same as before)
def warmup(warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return fn

# Removed setup_distributed and cleanup_distributed

def load_data(args, tokenizer):
    """Loads train and validation data from CSV files."""
    data_path = Path(args.data_dir)
    try:
        train_df = pd.read_csv(data_path / args.train_data)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {data_path / args.train_data}")
        raise
    try:
        valid_df = pd.read_csv(data_path / args.valid_data)
    except FileNotFoundError:
        print(f"Error: Validation data file not found at {data_path / args.valid_data}")
        print("Validation loop will be skipped if --valid_data is not provided or found.")
        valid_df = None # Allow running without validation

    # Ensure sequences and labels lists are populated here
    train_sequences = train_df['Sequence'].tolist()
    train_labels = train_df['label'].astype(int).tolist()

    if not train_sequences or not train_labels:
         raise ValueError("Please load your train sequences and labels!")
    if len(train_sequences) != len(train_labels):
        raise ValueError("Train sequences and labels must have the same length!")
    if not all(l > 0 for l in train_labels):
        raise ValueError("Train labels must be integers greater than 0!")

    max_label_train = max(train_labels) if train_labels else 0
    n_classes = max_label_train

    if valid_df is not None:
        valid_sequences = valid_df['Sequence'].tolist()
        valid_labels = valid_df['label'].astype(int).tolist()
        if not valid_sequences or not valid_labels:
             print("Warning: Validation file loaded but contains no sequences or labels.")
             valid_dataset = None
        elif len(valid_sequences) != len(valid_labels):
            raise ValueError("Validation sequences and labels must have the same length!")
        elif not all(l > 0 for l in valid_labels):
            raise ValueError("Validation labels must be integers greater than 0!")
        else:
            max_label_valid = max(valid_labels) if valid_labels else 0
            n_classes = max(n_classes, max_label_valid) # Update n_classes if validation has higher labels
            valid_dataset = ConditionalProteinDataset(
                valid_sequences,
                valid_labels,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len
            )
            print(f"Loaded {len(valid_sequences)} validation sequences.")
    else:
        valid_dataset = None
        print("No validation data loaded.")


    print(f"Loaded {len(train_sequences)} training sequences.")
    print(f"Detected {n_classes} classes based on data.")

    train_dataset = ConditionalProteinDataset(
        train_sequences,
        train_labels,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len
    )

    return train_dataset, valid_dataset, n_classes

def main():
    parser = argparse.ArgumentParser(description='Conditional D3PM Training (Single GPU)')

    # --- Paths and Data ---
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing data CSVs')
    parser.add_argument('--train_data', type=str, default='train_ec_311.csv', help='Filename for training data CSV')
    parser.add_argument('--valid_data', type=str, default='valid_ec_311.csv', help='Filename for validation data CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='./conditional_output_single', help='Directory for checkpoints and logs')

    # --- Model Hyperparameters ---
    parser.add_argument('--d_embed', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of ByteNet layers')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for ByteNet')
    parser.add_argument('--r', type=int, default=4, help='Dilation factor r')
    parser.add_argument('--class_dropout_prob', type=float, default=0.1, help='Dropout probability for class embedding (for CFG)')
    parser.add_argument('--embedding_scale', type=float, default=1.0, help='Scale factor for adding class embedding')

    # --- Diffusion Hyperparameters ---
    parser.add_argument('--diffusion_timesteps', type=int, default=50, help='Number of diffusion timesteps')
    parser.add_argument('--reweighting_term', type=float, default=0.01, help='Lambda weight for CE loss term in D3PM')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length (truncation/padding)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of linear warmup steps for LR scheduler')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients over N batches')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')

    # --- Logging and Checkpointing ---
    parser.add_argument('--log_freq', type=int, default=50, help='Log training metrics every N steps')
    parser.add_argument('--checkpoint_freq_steps', type=int, default=1000, help='Save checkpoint every N steps (set to 0 to disable step checkpoints)')
    parser.add_argument('--save_latest_only', action='store_true', help='Only keep the latest and best checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--log_to_file', action='store_true', help='Log metrics to CSV files in output_dir')

    # --- Other ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda:0", "cpu"). Auto-detects if None.')


    args = parser.parse_args()

    # Create output directory if it doesn't exist
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting single-GPU/CPU training with configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # --- Call the main training function ---
    train(args)

def train(args):
    """Main training function for single GPU/CPU."""

    # --- Device Selection ---
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed) # Seed all GPUs for consistency, though only one is used

    # --- Tokenizer and Diffusion Matrices ---
    tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, pad=PAD, all_aas=MSA_AAS, sequences=True)
    n_tokens = tokenizer.K + 2
    print(f"Tokenizer vocabulary size: {n_tokens}")

    Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=args.diffusion_timesteps)
    Q_t_gpu = Q_t.to(device) # Move to device for loss calculation
    Q_prod_gpu = Q_prod.to(device)
    print("Tokenizer and Diffusion matrices initialized.")

    # --- Load Data ---
    print("Loading data...")
    train_dataset, valid_dataset, n_classes = load_data(args, tokenizer)
    print("Data loading complete.")

    # --- Data Loaders ---
    # Removed DistributedSampler
    collater = ConditionalD3PMCollator(
        tokenizer=tokenizer,
        num_timesteps=args.diffusion_timesteps,
        Q=Q_t,      # Pass CPU tensors to collator
        Q_bar=Q_prod
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # Enable standard shuffling
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size, # Can use larger batch size for validation
            shuffle=False,
            collate_fn=collater,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    print("DataLoaders initialized.")

    # --- Model ---
    model = ConditionalByteNetLMTime(
        n_tokens=n_tokens,
        d_embedding=args.d_embed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        r=args.r,
        n_classes=n_classes,
        class_dropout_prob=args.class_dropout_prob,
        embedding_scale=args.embedding_scale,
        timesteps=args.diffusion_timesteps,
        padding_idx=tokenizer.pad_id
    ).to(device)
    # Removed SyncBatchNorm and DDP wrapping

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized on {device} with {num_params:,} trainable parameters.")

    # --- Optimizer, Scheduler, Loss, Scaler ---
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, warmup(args.warmup_steps))
    loss_func_lvb = D3PMLVBLoss(tmax=args.diffusion_timesteps, tokenizer=tokenizer).to(device)
    loss_func_ce = D3PMCELoss(tokenizer=tokenizer).to(device)
    _lambda = args.reweighting_term
    scaler = GradScaler(enabled=(device.type == 'cuda')) # Enable only for CUDA

    print("Optimizer, Scheduler, Loss Functions, Scaler initialized.")

    # --- Resume from Checkpoint ---
    start_epoch = 0
    total_steps = 0
    best_val_loss = float('inf')
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device) # Load directly to the target device

        model.load_state_dict(checkpoint['model_state_dict']) # Load state dict directly
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_steps = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'scaler_state_dict' in checkpoint and device.type == 'cuda':
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resumed from Epoch {start_epoch}, Step {total_steps}, Best Val Loss: {best_val_loss:.4f}")
    else:
        if args.resume_checkpoint:
            print(f"Warning: Checkpoint file not found at {args.resume_checkpoint}. Starting from scratch.")
        else:
            print("Starting training from scratch.")
        # Initialize log files if starting fresh and logging to file
        if args.log_to_file:
             log_dir = Path(args.output_dir)
             log_dir.mkdir(parents=True, exist_ok=True)
             with open(log_dir / 'train_metrics.csv', 'w') as f:
                 f.write('epoch,step,loss,nll_loss,lr\n')
             if valid_loader: # Only create validation log if validation data exists
                 with open(log_dir / 'valid_metrics.csv', 'w') as f:
                     f.write('epoch,step,loss,nll_loss\n')


    # --- Training Loop ---
    print(f"\n--- Starting Training from Epoch {start_epoch+1} ---")
    global_step = total_steps # Initialize global_step correctly when resuming
    for epoch in range(start_epoch, args.epochs):
        # Removed set_epoch for sampler

        model.train()
        epoch_loss_sum = 0.0
        epoch_nll_loss_sum = 0.0
        epoch_steps_logged = 0 # Track steps for averaging log printouts
        epoch_start_time = datetime.now()

        for i, batch in enumerate(train_loader):
            is_accumulation_step = (i + 1) % args.accumulate_grad_batches == 0
            effective_step = (i // args.accumulate_grad_batches) # Step counter for optimizer/scheduler updates
            current_global_step = global_step + effective_step + 1 # Track optimizer steps

            # --- Forward Pass ---
            src, src_onehot, timestep, tgt, tgt_onehot, _, _, _, class_labels = batch
            src, src_onehot, timestep, tgt, tgt_onehot, class_labels = \
                src.to(device), src_onehot.to(device), timestep.to(device), \
                tgt.to(device), tgt_onehot.to(device), class_labels.to(device)
            input_mask = (src != tokenizer.pad_id) # Shape: [B, L]

            # Use autocast for mixed precision if on GPU
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(src, timestep, class_labels=class_labels, input_mask=input_mask.unsqueeze(-1))

                # --- Loss Calculation ---
                lvb_loss = loss_func_lvb(src_onehot, None, outputs, tgt, tgt_onehot, input_mask, timestep, Q_t_gpu, Q_prod_gpu)
                ce_loss = loss_func_ce(outputs, tgt, input_mask)
                loss = lvb_loss + (_lambda * ce_loss)
                nll_loss = ce_loss # For logging

                # Scale loss for gradient accumulation
                loss_scaled = loss / args.accumulate_grad_batches

            # --- Backward Pass ---
            scaler.scale(loss_scaled).backward()

            # --- Optimization Step ---
            if is_accumulation_step:
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Zero gradients *after* optimizer step
                scheduler.step()      # Step scheduler *after* optimizer step

            # --- Accumulate Metrics for Logging ---
            # Use the *unscaled* loss for accurate logging
            epoch_loss_sum += loss.item()
            epoch_nll_loss_sum += nll_loss.item()
            epoch_steps_logged += 1

            # --- Logging ---
            if is_accumulation_step and current_global_step % args.log_freq == 0 :
                # Average loss over the logging frequency interval (approximately)
                avg_loss = epoch_loss_sum / epoch_steps_logged
                avg_nll = epoch_nll_loss_sum / epoch_steps_logged
                lr_current = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {current_global_step} | "
                      f"Batch {i+1}/{len(train_loader)} | "
                      f"Avg Loss: {avg_loss:.4f} | Avg NLL: {avg_nll:.4f} | " # Indicate these are averaged
                      f"LR: {lr_current:.2e}")
                # Log to file
                if args.log_to_file:
                    with open(os.path.join(args.output_dir, 'train_metrics.csv'), 'a') as f:
                        f.write(f"{epoch+1},{current_global_step},{avg_loss:.5f},{avg_nll:.5f},{lr_current:.6e}\n")
                # Reset epoch sums for next logging interval
                epoch_loss_sum = 0.0
                epoch_nll_loss_sum = 0.0
                epoch_steps_logged = 0

            # --- Checkpointing ---
            if args.checkpoint_freq_steps > 0 and is_accumulation_step and current_global_step % args.checkpoint_freq_steps == 0:
                 save_checkpoint(model, optimizer, scheduler, scaler, epoch, current_global_step, best_val_loss, args, filename=f"checkpoint_step_{current_global_step}.pt")

        # --- End of Epoch ---
        # Update total steps based on completed optimizer steps
        total_steps += len(train_loader) // args.accumulate_grad_batches
        epoch_duration = datetime.now() - epoch_start_time

        print("-" * 50)
        print(f"Epoch {epoch+1} Summary:")
        # Note: Final epoch avg loss isn't printed here unless log_freq aligns, could add it if needed
        print(f"  Completed Steps: {total_steps}")
        print(f"  Duration: {epoch_duration}")
        print("-" * 50)

        # --- Validation ---
        if valid_loader:
             val_loss, val_nll = run_validation(model, valid_loader, loss_func_lvb, loss_func_ce, _lambda, device, tokenizer, Q_t_gpu, Q_prod_gpu)
             print(f"  Validation Loss: {val_loss:.4f}")
             print(f"  Validation NLL:  {val_nll:.4f}")

             # Log validation metrics
             if args.log_to_file:
                 with open(os.path.join(args.output_dir, 'valid_metrics.csv'), 'a') as f:
                     f.write(f"{epoch+1},{total_steps},{val_loss:.5f},{val_nll:.5f}\n")

             # --- Save Checkpoint (Best and/or End of Epoch) ---
             is_best = val_loss < best_val_loss
             if is_best:
                 best_val_loss = val_loss
                 print(f"  New best validation loss: {best_val_loss:.4f}. Saving best checkpoint...")
                 save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_steps, best_val_loss, args, filename="checkpoint_best.pt")
        else:
             print("  Skipping validation (no validation data).")


        # Save latest checkpoint at end of every epoch
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_steps, best_val_loss, args, filename="checkpoint_latest.pt")

        print("-" * 50)


    print("Training finished.")
    # Removed cleanup_distributed

def run_validation(model, loader, loss_func_lvb, loss_func_ce, _lambda, device, tokenizer, Q_t_gpu, Q_prod_gpu):
    """Runs the validation loop."""
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_nll = 0.0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculations
        for i, batch in enumerate(loader):
            src, src_onehot, timestep, tgt, tgt_onehot, _, _, _, class_labels = batch
            src, src_onehot, timestep, tgt, tgt_onehot, class_labels = \
                src.to(device), src_onehot.to(device), timestep.to(device), \
                tgt.to(device), tgt_onehot.to(device), class_labels.to(device)
            input_mask = (src != tokenizer.pad_id)

            # Autocast can still be used for inference speedup
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(src, timestep, class_labels=class_labels, input_mask=input_mask.unsqueeze(-1))
                lvb_loss = loss_func_lvb(src_onehot, None, outputs, tgt, tgt_onehot, input_mask, timestep, Q_t_gpu, Q_prod_gpu)
                ce_loss = loss_func_ce(outputs, tgt, input_mask)
                loss = lvb_loss + (_lambda * ce_loss)
                nll_loss = ce_loss

            # Aggregate loss
            # Removed dist.all_reduce
            total_val_loss += loss.item()
            total_val_nll += nll_loss.item()
            num_batches += 1

    # Calculate average loss
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
    avg_val_nll = total_val_nll / num_batches if num_batches > 0 else 0.0

    return avg_val_loss, avg_val_nll # Return simple floats


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_val_loss, args, filename="checkpoint.pt"):
    """Saves a training checkpoint."""
    # Removed rank check
    if args.save_latest_only and "latest" not in filename and "best" not in filename:
        # Skip saving intermediate step checkpoints if only latest/best is desired
        # unless it's explicitly a step checkpoint (and freq > 0)
        if args.checkpoint_freq_steps > 0 and filename.startswith("checkpoint_step_"):
             pass # Allow saving step checkpoints even if latest_only is true
        else:
            return

    output_path = os.path.join(args.output_dir, filename)
    # Save model state directly (no DDP module)
    model_state_dict = model.state_dict()

    save_obj = {
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args) # Save args for reference
    }
    # Add scaler state dict if using CUDA
    if scaler.is_enabled(): # Check if scaler is enabled (i.e., on CUDA)
        save_obj['scaler_state_dict'] = scaler.state_dict()

    try:
        torch.save(save_obj, output_path)
        print(f"Checkpoint saved to {output_path}")
    except Exception as e:
        print(f"Error saving checkpoint to {output_path}: {e}")


    # Clean up old step checkpoints if saving latest only
    if args.save_latest_only and ("latest" in filename or "best" in filename): # Cleanup triggered by saving latest/best
         prefix = "checkpoint_step_"
         # Keep track of best to avoid deleting it
         best_ckpt_path = os.path.join(args.output_dir, "checkpoint_best.pt")
         steps_to_delete = [
             f for f in os.listdir(args.output_dir)
             if f.startswith(prefix) and f.endswith(".pt") and os.path.join(args.output_dir, f) != best_ckpt_path
         ]
         # Sort by step number potentially if needed for more complex cleanup rules
         for f_del in steps_to_delete:
             try:
                 os.remove(os.path.join(args.output_dir, f_del))
                 # print(f"Deleted old step checkpoint: {f_del}") # Optional: uncomment for verbose deletion
             except OSError as e:
                 print(f"Error deleting old checkpoint {f_del}: {e}")


if __name__ == '__main__':
    main()