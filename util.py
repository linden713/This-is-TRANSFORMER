import torch
import torch.nn as nn

def create_padding_mask(seq):
    """
    Create mask for padding tokens (0s)
    Args:
        seq: Input sequence tensor (batch_size, seq_len)
    Returns:
        mask: Padding mask (batch_size, 1, 1, seq_len)
    """
    batch_size, seq_len = seq.shape    
    output = torch.eq(seq, 0)  # Create a boolean mask where 0s are True
    return output.view(batch_size, 1, 1, seq_len)

def create_future_mask(size):
    """
    Create mask to prevent attention to future positions
    Args:
        size: Size of square mask (target_seq_len)
    Returns:
        mask: Future mask (1, 1, size, size)
    """
    # Create upper triangular matrix and invert it
    mask = torch.triu(torch.ones(1, 1, size, size), diagonal=1)==0
    return mask

def create_masks(src, tgt):
    """
    Create all masks needed for training
    Args:
        src: Source sequence (batch_size, src_len)
        tgt: Target sequence (batch_size, tgt_len)
    Returns:
        src_mask: Padding mask for encoder
        tgt_mask: Combined padding and future mask for decoder
    """
    # 1. Create padding masks
    src_padding_mask = create_padding_mask(src)
    tgt_padding_mask = create_padding_mask(tgt)
    # 2. Create future mask
    tgt_len = tgt.size(1)
    tgt_future_mask = create_future_mask(tgt_len).to(tgt.device) #TODO 

    return src_padding_mask, tgt_padding_mask & tgt_future_mask



    

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Args:
            optimizer: Optimizer to adjust learning rate for
            d_model: Model dimensionality
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps


    def step(self, step_num):
        """
        Update learning rate based on step number
        """

        # lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        learning_rate = self.d_model ** -0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        Args:
            logits: Model predictions (batch_size, vocab_size) #each row of vocab_size contains probability score of each label
            target: True labels (batch_size) #each row of batch size contains the index to the correct label
        """
        # vocab_size = logits.vocab_size(-1)
        vocab_size = logits.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1),self.confidence)
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(logits, dim=-1), dim=-1))