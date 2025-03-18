import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """
    Applies RoPE to Q or K. 
    Expects shape [batch, seq_len, num_heads, head_dim].
    We rotate pairs: (0,1), (2,3), ...
    """
    def __init__(self, base=10000.0):
        super().__init__()
        self.base = base

    def forward(self, x):
        # x shape: [batch, seq_len, num_heads, head_dim]
        # We rotate in the last dimension in pairs.
        bsz, seq_len, nh, hd = x.shape
        half = hd // 2
        # We'll build position indices [0..seq_len-1]
        pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(-1)  # [seq_len, 1]
        
        # For frequency we do base^(-2*j/hd)
        # j in [0..half-1], shape [1..half]
        freq_seq = torch.arange(half, dtype=x.dtype, device=x.device)
        freq = (self.base ** (-2.0 * freq_seq / hd)).unsqueeze(0)  # [1, half]
        
        # angle = pos * freq => shape [seq_len, half]
        angle = pos * freq  # broadcast: [seq_len, half]
        
        # sin, cos => shape [seq_len, half]
        sin = angle.sin().unsqueeze(-1)  # [seq_len, half, 1]
        cos = angle.cos().unsqueeze(-1)  # [seq_len, half, 1]

        # Fix: Use reshape instead of view for non-contiguous tensors
        sin = sin.permute(1,0,2).reshape(1, seq_len, 1, half)
        cos = cos.permute(1,0,2).reshape(1, seq_len, 1, half)

        # x has shape [bsz, seq_len, nh, hd]
        # We'll slice x into (x0, x1) pairs along the last dim
        x0 = x[..., 0::2]  # shape [bsz, seq_len, nh, half]
        x1 = x[..., 1::2]  # shape [bsz, seq_len, nh, half]

        # We need to do the rotation:
        #   rx0 = x0*cos - x1*sin
        #   rx1 = x1*cos + x0*sin
        # We'll reshape sin, cos to broadcast:
        # sin, cos => [1, seq_len, 1, half]
        
        #sin = sin.permute(1,0,2).view(1, seq_len, 1, half)
        #cos = cos.permute(1,0,2).view(1, seq_len, 1, half)

        rx0 = x0*cos - x1*sin
        rx1 = x1*cos + x0*sin

        # interleave them back on the last dim
        # final shape is [bsz, seq_len, nh, hd]
        out = torch.zeros_like(x)
        out[..., 0::2] = rx0
        out[..., 1::2] = rx1
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        # Output projection
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding() if rope else None

    def forward(self, x):
        """
        x shape: [batch, seq_len, d_model]
        Return: [batch, seq_len, d_model]
        """
        bsz, seq_len, _ = x.size()
        # Q, K, V => [batch, seq_len, d_model]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Reshape into heads => [bsz, seq_len, num_heads, head_dim]
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K if desired
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        # Attention: Q*K^T / sqrt(head_dim)
        # We'll do the batch matrix multiply approach:
        # Q => [bsz, seq_len, nh, hd]
        # K => [bsz, seq_len, nh, hd]
        # We want attn_logits => [bsz, nh, seq_len, seq_len]
        Q_ = Q.permute(0,2,1,3)  # [bsz, nh, seq_len, hd]
        K_ = K.permute(0,2,1,3)  # [bsz, nh, seq_len, hd]

        # attn_logits = Q_ @ K_.transpose(-2, -1)
        attn_logits = torch.matmul(Q_, K_.transpose(-2, -1))  # shape [bsz, nh, seq_len, seq_len]
        attn_logits /= math.sqrt(self.head_dim)

        # softmax
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [bsz, nh, seq_len, seq_len]

        # Weighted sum of V
        V_ = V.permute(0,2,1,3)  # [bsz, nh, seq_len, hd]
        out_ = torch.matmul(attn_weights, V_)  # [bsz, nh, seq_len, hd]

        # re‐permute: [bsz, seq_len, nh, hd]
        out = out_.permute(0,2,1,3).contiguous()  # [bsz, seq_len, nh, hd] => [bsz, seq_len, d_model]
        out = out.view(bsz, seq_len, self.d_model)

        # final projection
        out = self.Wo(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    For cross-attention: Q from X, K/V from M.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Qx, Mx):
        """
        Qx shape: [bsz, LQ, d_model]
        Mx shape: [bsz, LM, d_model] used for K and V
        returns: [bsz, LQ, d_model]
        """
        bsz, LQ, _ = Qx.size()
        LM = Mx.size(1)

        Q = self.Wq(Qx)  # [bsz, LQ, d_model]
        K = self.Wk(Mx)  # [bsz, LM, d_model]
        V = self.Wv(Mx)  # [bsz, LM, d_model]

        # reshape
        Q = Q.view(bsz, LQ, self.num_heads, self.head_dim).permute(0,2,1,3)  # [bsz, nh, LQ, hd]
        K = K.view(bsz, LM, self.num_heads, self.head_dim).permute(0,2,1,3)  # [bsz, nh, LM, hd]
        V = V.view(bsz, LM, self.num_heads, self.head_dim).permute(0,2,1,3)  # [bsz, nh, LM, hd]

        # attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1))  # [bsz, nh, LQ, LM]
        attn_logits /= math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [bsz, nh, LQ, LM]
        out_ = torch.matmul(attn_weights, V)  # [bsz, nh, LQ, hd]

        # reshape back
        out_ = out_.permute(0,2,1,3).contiguous()  # [bsz, LQ, nh, hd]
        out = out_.view(bsz, LQ, self.d_model)
        out = self.Wo(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
        )
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    One encoder layer with:
      Y = LN(X + SelfAttn(X))
      Z = LN(Y + FF(Y))
    """
    def __init__(self, d_model, ff_hidden_dim, num_heads, rope=True):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x):
        # x: [bsz, seq_len, d_model]
        sa = self.self_attn(x)
        x = self.ln1(x + sa)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class Encoder2Layer(nn.Module):
    """
    One layer for the second encoder with cross-attention:
      Y = LN(X + CrossAttn(X, M))  # M is memory from first encoder
      Z = LN(Y + FF(Y))
    """
    def __init__(self, d_model, ff_hidden_dim, num_heads):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = FeedForward(d_model, ff_hidden_dim)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, mem):
        # cross attn
        ca = self.cross_attn(x, mem)
        x = self.ln1(x + ca)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class TwoEncoderTransformer(nn.Module):
    def __init__(self, 
                 D_in,       # input numeric feature dim for first encoder (excluding libcell embedding) e.g. 18
                 D_out,      # output feature dim (max number of classes)
                 D_emb,      # embedding size for libcell and libcell_type, e.g. 768
                 D_model,    # main hidden size
                 FF_hidden_dim, 
                 num_heads, 
                 num_encoder_layers, 
                 num_encoder_layers_2,
                 libcell_embedding, # torch 2D tensor (libcell_num, D_emb), used to load embeddings to encoder_1 input
                 libcell_type_embedding # torch 2D tensor (libcell_type_num, D_emb) used to load embeddings to encoder_2 input
                 ):
        super().__init__()
        self.D_in = D_in
        self.D_emb = D_emb
        self.D_model = D_model

        # Instantiate embeddings
        self.libcell_embedding_table = nn.Embedding.from_pretrained(libcell_embedding, freeze=True, max_norm=None, padding_idx=0)
        self.libcell_type_embedding_table = nn.Embedding.from_pretrained(libcell_type_embedding, freeze=True, max_norm=None, padding_idx=0)

        # Projection in
        self.proj_in1 = nn.Linear(D_in + D_emb, D_model, bias=False)
        self.proj_in2 = nn.Linear(D_emb, D_model, bias=False)
        # final out
        self.proj_out = nn.Linear(D_model, D_out, bias=False)

        # first encoder layers
        self.encoders1 = nn.ModuleList([
            EncoderLayer(D_model, FF_hidden_dim, num_heads, rope=True)
            for _ in range(num_encoder_layers)
        ])

        # second encoder layers
        self.encoders2 = nn.ModuleList([
            Encoder2Layer(D_model, FF_hidden_dim, num_heads)
            for _ in range(num_encoder_layers_2)
        ])

    def forward(self, encoder_1_numeric_data, encoder_1_libcell_ids, encoder_2_libcell_type_ids):
        """
        encoder_1_numeric_data: shape [bsz, L, D_in], dtype: torch.float32
        encoder_1_libcell_ids: shape [bsz, L], dtype: torch.int64
        encoder_2_libcell_type_ids: shape [bsz, L2], dtype: torch.int64
        returns: [bsz, L2, D_out]
        """

        # Embedding lookup, if -1 is found in libcell_ids, it will be replaced with 0 so that padding_idx is used
        encoder_1_libcell_id_mask = (encoder_1_libcell_ids == -1)
        encoder_1_libcell_ids = encoder_1_libcell_ids.masked_fill(encoder_1_libcell_id_mask, 0)
        libcell_emb = self.libcell_embedding_table(encoder_1_libcell_ids)
        data1 = torch.cat([encoder_1_numeric_data, libcell_emb], dim=-1)  # => [bsz, L, D_in + D_emb]

        encoder_2_libcell_type_id_mask = (encoder_2_libcell_type_ids == -1)
        encoder_2_libcell_type_ids = encoder_2_libcell_type_ids.masked_fill(encoder_2_libcell_type_id_mask, 0)
        libcell_type_emb = self.libcell_type_embedding_table(encoder_2_libcell_type_ids)
        data2 = libcell_type_emb  # => [bsz, L2, D_emb]


        # First encoder
        x1 = self.proj_in1(data1)  # => [bsz, L, D_model]
        for layer in self.encoders1:
            x1 = layer(x1)  # [bsz, L, D_model]

        # Second encoder
        x2 = self.proj_in2(data2)  # => [bsz, L2, D_model]
        for layer2 in self.encoders2:
            x2 = layer2(x2, x1)     # cross-attend to x1

        # final projection => [bsz, L2, D_out]
        out = self.proj_out(x2)

        # Apply softmax to get probabilities
        out = nn.functional.softmax(out, dim=-1)
        return out

import struct
import numpy as np

def export_model_weights_robust(model: torch.nn.Module, filename: str):
    """
    Export weights with the following format:
      1) param_count: size_t
      For each param p:
        a) name_length: size_t
        b) name bytes
        c) ndims: size_t
        d) dims[ndims]: size_t array
        e) float data in row-major order
    """
    # Extract state_dict as (name, tensor) pairs
    state_dict = model.state_dict()
    params = list(state_dict.items())  # [(name, tensor), ...]

    with open(filename, "wb") as f:
        # 1) param_count
        param_count = len(params)
        f.write(struct.pack("Q", param_count))  # 'Q' for 64-bit unsigned (size_t on many platforms)

        for name, tensor in params:
            arr = tensor.detach().cpu().numpy().astype(np.float32)

            # a) name_length
            name_bytes = name.encode("utf-8")
            name_len = len(name_bytes)
            f.write(struct.pack("Q", name_len))

            # b) name
            f.write(name_bytes)

            # c) ndims
            ndims = arr.ndim
            f.write(struct.pack("Q", ndims))

            # d) dims
            for dim_size in arr.shape:
                f.write(struct.pack("Q", dim_size))

            # e) data (row-major)
            f.write(arr.tobytes())

    print(f"Exported {param_count} parameters to {filename}")
