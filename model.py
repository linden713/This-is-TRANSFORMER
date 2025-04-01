import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from util import create_masks

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_layers=6,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,d_ff,dropout,max_seq_length)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,d_ff,dropout,max_seq_length)
        
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size )
        
    def forward(self, src, tgt):
        mask_src, mask_tgt = create_masks(src, tgt)
        
        encoder_output = self.encoder(src, mask_src)
        decoder_output = self.decoder(tgt,encoder_output, mask_src, mask_tgt)
        result = self.projection_layer(decoder_output)
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def scaled_dot_product_attention(query,key,value,mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask to prevent attention to certain positions
        """
        assert query.dim() ==4, f"query dim: {query.dim()}"
        assert key.size(-1) == query.size(-1), f"key dim: {key.size(-1)} query dim: {query.size(-1)}"
        assert value.size(-1) == query.size(-1), f"value dim: {value.size(-1)} query dim: {query.size(-1)}"
        d_k = query.size(-1)
        
        score = torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(d_k)
        if mask is not None:
            score = score.mask_fill(mask ==0, float('-inf'))
            
        attention_weight = softmax(score, dim =-1)
        return attention_weight.matmul(value)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        sequence_len = query.size[1] 
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = Q.view(batch_size, sequence_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, d_k)
        K = K.view(batch_size, sequence_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, d_k)
        V = V.view(batch_size, sequence_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_v, d_k)
        
        attention_weight = self.scaled_dot_product_attention(Q, K, V, mask)

        output = attention_weight.transpose(1,2).contiguous().view(batch_size, sequence_len, self.d_model)
        
        return self.W_o(output)
    
class FeedforwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network
    Args:
        d_model: input/output dimension
        d_ff: hidden dimension
        dropout: dropout rate (default=0.1)
        Two way to implement: MPL or CNN
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length,d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (max_seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term) # Shape: (max_seq_length, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, d_model)
    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)] 
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardNetwork(d_model, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for padding
        Returns:
            x: Output tensor of shape (batch_size, seq_len, d_model)
        """    
        x = x + self.self_attn(x,x,x,mask)
        x = self.dropout(x)
        x = self.layer_norm1(x)
        
        x_ff = x + self.feedforward(x)
        x = self.dropout(x_ff)
        x = self.layer_norm2(x)
        return x
            
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardNetwork(d_model, d_ff, dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence embedding (batch_size, target_seq_len, d_model)
            encoder_output: Output from encoder (batch_size, source_seq_len, d_model)
            src_mask: Mask for source padding
            tgt_mask: Mask for target padding and future positions
        """
        # Self-attention on target sequence
        x = x + self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.layer_norm1(x)
        
        # Encoder-decoder attention
        x = x + self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x = self.layer_norm2(x)
        
        # Feedforward network
        x_ff = x + self.feedforward(x)
        x = self.dropout(x_ff)
        x = self.layer_norm3(x)
        
        return x
        
class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_layers=6,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tokens (batch_size, seq_len)
            mask: Mask for padding positions
        Returns:
            encoder_output: (batch_size, seq_len, d_model)
        """
        x = self.embedding(x) * self.scale  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.dropout(x)  # Apply dropout
        for layer in self.encoder_layers:
            x= layer(x, mask)  # Pass through encoder layers
        return x

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_layers=6,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target tokens (batch_size, target_seq_len)
            encoder_output: Output from encoder (batch_size, source_seq_len, d_model)
            src_mask: Mask for source padding
            tgt_mask: Mask for target padding and future positions
        Returns:
            decoder_output: (batch_size, target_seq_len, d_model)
        """
        x = self.embedding(x) * self.scale
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x