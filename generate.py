import torch

def generate_conditional_d3pm(
    model,
    tokenizer,
    Q,
    Q_bar,
    timesteps,
    seq_len,
    class_labels,
    guidance_scale=3.0,
    batch_size=1,
    device='cuda'
):
    """
    Generate sequences using conditional D3PM with classifier-free guidance.
    
    Args:
        model: ConditionalByteNetLMTime model
        tokenizer: Tokenizer instance
        Q: Transition matrix
        Q_bar: Cumulative product of transition matrices
        timesteps: Total number of diffusion timesteps
        seq_len: Length of sequences to generate
        class_labels: Class labels for conditioning (must be >= 1, int tensor)
        guidance_scale: Strength of classifier-free guidance (higher = stronger conditioning)
        batch_size: Number of sequences to generate in parallel
        device: Device to run generation on
        
    Returns:
        tuple: (token_sequences, untokenized_sequences)
    """
    model.eval()
    
    # Start with random tokens (initial noise)
    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long).to(device)
    
    # Move matrices to device
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)
    
    # Ensure class_labels is a tensor with the right shape
    if not isinstance(class_labels, torch.Tensor):
        class_labels = torch.tensor([class_labels] * batch_size, device=device)
    elif len(class_labels.shape) == 0:
        class_labels = torch.tensor([class_labels.item()] * batch_size, device=device)
    elif len(class_labels) != batch_size:
        class_labels = class_labels.repeat(batch_size)
    class_labels = class_labels.to(device)
    
    # Reverse diffusion process - iterate backwards through timesteps
    reverse_timesteps = torch.linspace(timesteps-1, 1, timesteps-1, dtype=int)
    reverse_timesteps = reverse_timesteps.to(device)
    
    with torch.no_grad():
        for t in reverse_timesteps:
            # Create timestep tensor
            t_batch = torch.tensor([t] * batch_size, device=device)
            
            # Get model prediction with classifier-free guidance
            guided_pred = model.sample(
                sample, 
                t_batch, 
                class_labels=class_labels,
                guidance_scale=guidance_scale
            )
            
            # Convert to probabilities
            p = guided_pred[:, :, :tokenizer.K]  # Only predict standard AAs
            p = torch.nn.functional.softmax(p, dim=-1)
            p = p.to(torch.float64)
            
            # Prepare for next timestep
            x_tminus1 = sample.clone()
            
            # Sample next tokens
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred)  # [P x K x K]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), p[i].unsqueeze(2)).squeeze()
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                
                # Sample from the predicted distribution
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                
                # On final timestep, restrict to standard AAs
                if t == 1:
                    x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K-6], num_samples=1).squeeze()
            
            # Update sample for next iteration
            sample = x_tminus1
    
    # Convert tokens to strings
    untokenized = [tokenizer.untokenize(s) for s in sample]
    
    return sample, untokenized