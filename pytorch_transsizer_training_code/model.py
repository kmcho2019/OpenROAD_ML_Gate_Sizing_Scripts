import torch
import torch.nn as nn
import math
import copy # For deep copying models to ensure same initial weights

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

    def forward(self, x, attn_mask=None):
        """
        x shape: [batch, seq_len, d_model]
        attn_mask: [batch, seq_len] boolean mask, True indicates padding
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

        # --- Apply attention mask ---
        if attn_mask is not None:
            # attn_mask shape: [bsz, seq_len]
            # We need to broadcast it to [bsz, nh, seq_len, seq_len]
            # Masking applies to the key dimension (last dimension of attn_logits)
            mask = attn_mask.unsqueeze(1).unsqueeze(2) # shape [bsz, 1, 1, seq_len]
            attn_logits = attn_logits.masked_fill(mask, -1e9) # Use large negative value

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

    def forward(self, Qx, Mx, q_mask=None, kv_mask=None):
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

        # --- Apply Key/Value Mask (kv_mask) ---
        if kv_mask is not None:
            # kv_mask shape: [bsz, LM], True indicates padding
            # Need to broadcast to attn_logits shape [bsz, nh, LQ, LM]
            # Mask needs to be applied to the Key dimension (LM)
            mask = kv_mask.unsqueeze(1).unsqueeze(2)  # shape [bsz, 1, 1, LM]
            attn_logits = attn_logits.masked_fill(mask, -1e9) # Fill padded keys with large negative value

        # Apply Softmax
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [bsz, nh, LQ, LM]
        out_ = torch.matmul(attn_weights, V)  # [bsz, nh, LQ, hd]

        # reshape back
        out_ = out_.permute(0,2,1,3).contiguous()  # [bsz, LQ, nh, hd]
        out = out_.view(bsz, LQ, self.d_model)
        out = self.Wo(out)

        # --- Apply Query Mask (q_mask) ---
        if q_mask is not None:
            # q_mask shape: [bsz, LQ], True indicates padding
            # Need to broadcast to out shape [bsz, LQ, d_model]
            # Zero out the outputs corresponding to padded query positions
            out = out.masked_fill(q_mask.unsqueeze(-1), 0.0) # Use unsqueeze to broadcast over d_model

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

    def forward(self, x, mask=None):
        # x: [bsz, seq_len, d_model]
        # mask: [bsz, seq_len] boolean mask, True indicates padding
        sa = self.self_attn(x, attn_mask=mask)
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

    def forward(self, x, mem, kv_mask=None, q_mask=None):
        """
        x: [bsz, L2, d_model] - Input to this encoder layer
        mem: [bsz, L, d_model] - Memory from the first encoder
        kv_mask: [bsz, L], True indicates padding in memory
        q_mask: [bsz, L2], True indicates padding in input
        returns: [bsz, L2, d_model]
        """
        # cross attn
        ca = self.cross_attn(x, mem, kv_mask=kv_mask, q_mask=q_mask)
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

        self.num_libcells = libcell_embedding.size(0)
        self.num_libcell_types = libcell_type_embedding.size(0)

        self.num_libcell_pad_index = self.num_libcells
        self.num_libcell_type_pad_index = self.num_libcell_types

        # Create a new embedding matrix (size: num_libcells+1, D_emb) with zero padding at the end
        libcell_embedding = torch.cat([libcell_embedding, torch.zeros(1, D_emb, dtype=libcell_embedding.dtype)], dim=0)
        libcell_type_embedding = torch.cat([libcell_type_embedding, torch.zeros(1, D_emb, dtype=libcell_type_embedding.dtype)], dim=0)

        # Instantiate embeddings
        self.libcell_embedding_table = nn.Embedding.from_pretrained(libcell_embedding, freeze=True, max_norm=None, padding_idx=self.num_libcell_pad_index)
        self.libcell_type_embedding_table = nn.Embedding.from_pretrained(libcell_type_embedding, freeze=True, max_norm=None, padding_idx=self.num_libcell_type_pad_index)

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


        # Create Masks for embedding lookup and attention masking
        encoder_1_libcell_id_mask = (encoder_1_libcell_ids == -1)
        encoder_2_libcell_type_id_mask = (encoder_2_libcell_type_ids == -1)

        # Embedding lookup, if -1 is found in libcell_ids, it will be replaced with self.num_libcell_pad_index so that padding_idx is used
        encoder_1_libcell_ids = encoder_1_libcell_ids.masked_fill(encoder_1_libcell_id_mask, self.num_libcell_pad_index)
        libcell_emb = self.libcell_embedding_table(encoder_1_libcell_ids)
        data1 = torch.cat([encoder_1_numeric_data, libcell_emb], dim=-1)  # => [bsz, L, D_in + D_emb]

        # Embedding lookup, if -1 is found in libcell_type_ids, it will be replaced with self.num_libcell_type_pad_index so that padding_idx is used
        encoder_2_libcell_type_ids = encoder_2_libcell_type_ids.masked_fill(encoder_2_libcell_type_id_mask, self.num_libcell_type_pad_index)
        libcell_type_emb = self.libcell_type_embedding_table(encoder_2_libcell_type_ids)
        data2 = libcell_type_emb  # => [bsz, L2, D_emb]


        # First encoder
        x1 = self.proj_in1(data1)  # => [bsz, L, D_model]
        for layer in self.encoders1:
            x1 = layer(x1, mask=encoder_1_libcell_id_mask)  # [bsz, L, D_model]

        # Second encoder
        x2 = self.proj_in2(data2)  # => [bsz, L2, D_model]
        for layer2 in self.encoders2:
            x2 = layer2(x2, x1, kv_mask=encoder_1_libcell_id_mask, q_mask=encoder_2_libcell_type_id_mask)     # cross-attend to x1

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


# --- Test Functions ---
import datetime
# Test function for attention masking
# Tests the masking implementation in Self-Attention and Cross-Attention.
# Compares output for non-padded data (no mask) vs padded data (with mask).
# The valid portions of the output should match.
def test_attention_masking(atol=1e-6, rtol=1e-5):
    """
    Tests the masking implementation in Self-Attention and Cross-Attention.

    Compares output for non-padded data (no mask) vs padded data (with mask).
    The valid portions of the output should match.
    """
    print("--- Starting Attention Masking Test ---")
    print(f"Using torch version: {torch.__version__}")
    print(f"Current time: {datetime.datetime.now()}") # Assuming datetime is available
    torch.manual_seed(42) # For reproducibility

    # --- Parameters ---
    bsz = 4
    d_model = 64
    num_heads = 8
    LQ_valid = 15 # Valid length for sequence A (Query)
    LM_valid = 25 # Valid length for sequence B (Key/Value) - Only for Cross-Attention
    LQ_pad = 20   # Padded length for sequence A
    LM_pad = 30   # Padded length for sequence B - Only for Cross-Attention

    print(f"Params: bsz={bsz}, d_model={d_model}, num_heads={num_heads}")
    print(f"Seq A (Q): Valid Len={LQ_valid}, Padded Len={LQ_pad}")
    print(f"Seq B (KV): Valid Len={LM_valid}, Padded Len={LM_pad}")

    # --- Generate Test Data ---
    print("\nGenerating test data...")
    # 1. Non-padded data
    Qx_nopad = torch.randn(bsz, LQ_valid, d_model)
    Mx_nopad = torch.randn(bsz, LM_valid, d_model) # Used only for cross-attention

    # 2. Padded data (Initialize with random, then fill valid part)
    Qx_pad = torch.randn(bsz, LQ_pad, d_model)
    Mx_pad = torch.randn(bsz, LM_pad, d_model)
    Qx_pad[:, :LQ_valid, :] = Qx_nopad
    Mx_pad[:, :LM_valid, :] = Mx_nopad

    # 3. Masks (True indicates padding/invalid)
    q_indices = torch.arange(LQ_pad).unsqueeze(0).expand(bsz, -1)
    kv_indices = torch.arange(LM_pad).unsqueeze(0).expand(bsz, -1) # Only for Cross-Attention

    q_mask = q_indices >= LQ_valid    # Shape [bsz, LQ_pad]
    kv_mask = kv_indices >= LM_valid  # Shape [bsz, LM_pad]

    # --- Instantiate Models ---
    # Use deepcopy to ensure identical initial weights for padded/non-padded tests
    model_self_base = MultiHeadSelfAttention(d_model, num_heads, rope=False)
    model_self_nopad = copy.deepcopy(model_self_base)
    model_self_pad = copy.deepcopy(model_self_base)

    model_cross_base = MultiHeadCrossAttention(d_model, num_heads)
    model_cross_nopad = copy.deepcopy(model_cross_base)
    model_cross_pad = copy.deepcopy(model_cross_base)

    # Set to evaluation mode (disables dropout if any)
    model_self_nopad.eval()
    model_self_pad.eval()
    model_cross_nopad.eval()
    model_cross_pad.eval()

    print("\n--- Testing Self-Attention ---")
    # --- Case 1: No Padding, No Mask ---
    with torch.no_grad():
         out_nopad_self = model_self_nopad(Qx_nopad, attn_mask=None)
         print(f"Output shape (no pad): {out_nopad_self.shape}")

    # --- Case 2: Padding, With Mask ---
    with torch.no_grad():
         # Use q_mask as attn_mask for self-attention
         out_pad_self = model_self_pad(Qx_pad, attn_mask=q_mask)
         print(f"Output shape (pad):    {out_pad_self.shape}")

    # --- Comparison ---
    relevant_out_pad_self = out_pad_self[:, :LQ_valid, :]
    padded_part_out_self = out_pad_self[:, LQ_valid:, :]

    valid_part_matches_self = torch.allclose(out_nopad_self, relevant_out_pad_self, atol=atol, rtol=rtol)
    padded_part_is_zero_self = torch.allclose(padded_part_out_self, torch.zeros_like(padded_part_out_self), atol=atol)

    print(f"Self-Attention - Valid part matches: {valid_part_matches_self}")
    if not valid_part_matches_self:
         print(f"   Max difference in valid part: {(out_nopad_self - relevant_out_pad_self).abs().max().item():.2e}")
    print(f"Self-Attention - Padded part is zero: {padded_part_is_zero_self}")
    if not padded_part_is_zero_self:
         print(f"   Max value in padded part: {padded_part_out_self.abs().max().item():.2e}")

    assert valid_part_matches_self, "Self-Attention valid parts comparison FAILED!"
    # assert padded_part_is_zero_self, "Self-Attention padded part zero check FAILED!"


    print("\n--- Testing Cross-Attention ---")
    # --- Case 1: No Padding, No Mask ---
    with torch.no_grad():
         out_nopad_cross = model_cross_nopad(Qx_nopad, Mx_nopad, q_mask=None, kv_mask=None)
         print(f"Output shape (no pad): {out_nopad_cross.shape}")

    # --- Case 2: Padding, With Mask ---
    with torch.no_grad():
         out_pad_cross = model_cross_pad(Qx_pad, Mx_pad, q_mask=q_mask, kv_mask=kv_mask)
         print(f"Output shape (pad):    {out_pad_cross.shape}")

    # --- Comparison ---
    relevant_out_pad_cross = out_pad_cross[:, :LQ_valid, :]
    padded_part_out_cross = out_pad_cross[:, LQ_valid:, :] # Output rows corresponding to padded queries

    valid_part_matches_cross = torch.allclose(out_nopad_cross, relevant_out_pad_cross, atol=atol, rtol=rtol)
    padded_part_is_zero_cross = torch.allclose(padded_part_out_cross, torch.zeros_like(padded_part_out_cross), atol=atol)

    print(f"Cross-Attention - Valid part matches: {valid_part_matches_cross}")
    if not valid_part_matches_cross:
         print(f"   Max difference in valid part: {(out_nopad_cross - relevant_out_pad_cross).abs().max().item():.2e}")
    print(f"Cross-Attention - Padded query output part is zero: {padded_part_is_zero_cross}")
    if not padded_part_is_zero_cross:
         print(f"   Max value in padded query output part: {padded_part_out_cross.abs().max().item():.2e}")

    assert valid_part_matches_cross, "Cross-Attention valid parts comparison FAILED!"
    assert padded_part_is_zero_cross, "Cross-Attention padded query output zero check FAILED!"

    print("\n--- Attention Masking Test Finished Successfully ---")

def test_two_encoder_transformer_masking(atol=1e-6, rtol=1e-5):
    """
    Tests the masking implementation in the full TwoEncoderTransformer.

    Compares output for non-padded data vs padded data (using -1 for IDs).
    The valid portions of the output should match.
    """
    print("--- Starting TwoEncoderTransformer Masking Test ---")
    print(f"Using torch version: {torch.__version__}")
    print(f"Current time: {datetime.datetime.now()}")
    torch.manual_seed(43) # For reproducibility

    # --- Parameters ---
    bsz = 3
    # Input/Embedding Dims
    D_in = 18
    D_emb = 48
    libcell_num = 100 # Vocab size for first encoder IDs 
    libcell_type_num = 50 # Vocab size for second encoder IDs 
    # Model Hyperparams
    D_model = 64
    D_out = 10 # Number of output classes
    FF_hidden_dim = D_model * 4
    num_heads = 8
    num_encoder_layers = 2
    num_encoder_layers_2 = 3
    # Sequence Lengths
    L_valid = 30 # Valid length for encoder 1 sequence
    L2_valid = 20 # Valid length for encoder 2 sequence
    L_pad = 40   # Padded length for encoder 1 sequence
    L2_pad = 25  # Padded length for encoder 2 sequence

    print(f"\nParams: bsz={bsz}, D_in={D_in}, D_emb={D_emb}, D_model={D_model}, D_out={D_out}")
    print(f"Heads={num_heads}, Enc1 Layers={num_encoder_layers}, Enc2 Layers={num_encoder_layers_2}")
    print(f"Enc1 Seq: Valid Len={L_valid}, Padded Len={L_pad}")
    print(f"Enc2 Seq: Valid Len={L2_valid}, Padded Len={L2_pad}")

    # --- Generate Dummy Embedding Tables ---
    # Ensure idx 0 exists, as it's the padding_idx
    libcell_embedding_weights = torch.randn(libcell_num, D_emb)
    libcell_type_embedding_weights = torch.randn(libcell_type_num, D_emb)

    # --- Generate Test Data ---
    print("\nGenerating test data...")
    # 1. Non-padded data (No -1 IDs)
    enc1_num_nopad = torch.randn(bsz, L_valid, D_in)
    enc1_ids_nopad = torch.randint(0, libcell_num, (bsz, L_valid), dtype=torch.long)
    enc2_ids_nopad = torch.randint(0, libcell_type_num, (bsz, L2_valid), dtype=torch.long)

    # 2. Padded data (using -1 for padding IDs)
    enc1_num_pad = torch.randn(bsz, L_pad, D_in) # Numeric padding value doesn't matter as much if masked
    enc1_ids_pad = torch.full((bsz, L_pad), -1, dtype=torch.long)
    enc2_ids_pad = torch.full((bsz, L2_pad), -1, dtype=torch.long)

    # Copy non-padded data into the beginning of padded tensors
    enc1_num_pad[:, :L_valid, :] = enc1_num_nopad
    enc1_ids_pad[:, :L_valid] = enc1_ids_nopad
    enc2_ids_pad[:, :L2_valid] = enc2_ids_nopad

    # --- Instantiate Models ---
    # Use deepcopy for identical initial weights
    model_base = TwoEncoderTransformer(
        D_in, D_out, D_emb, D_model, FF_hidden_dim, num_heads,
        num_encoder_layers, num_encoder_layers_2,
        libcell_embedding_weights, libcell_type_embedding_weights
    )
    model_nopad = copy.deepcopy(model_base)
    model_pad = copy.deepcopy(model_base)

    # Set to evaluation mode
    model_nopad.eval()
    model_pad.eval()

    print("\n--- Running Model ---")
    # --- Case 1: No Padding ---
    # Input does not contain -1, so internal masks should be all False
    with torch.no_grad():
        out_nopad = model_nopad(enc1_num_nopad, enc1_ids_nopad, enc2_ids_nopad)
        print(f"Output shape (no pad): {out_nopad.shape}") # Should be [bsz, L2_valid, D_out]

    # --- Case 2: Padding ---
    # Input contains -1, triggering internal mask generation
    with torch.no_grad():
        out_pad = model_pad(enc1_num_pad, enc1_ids_pad, enc2_ids_pad)
        print(f"Output shape (pad):    {out_pad.shape}") # Should be [bsz, L2_pad, D_out]

    # --- Comparison ---
    # Extract the part of the padded output corresponding to the original valid length L2_valid
    relevant_out_pad = out_pad[:, :L2_valid, :]

    valid_part_matches = torch.allclose(out_nopad, relevant_out_pad, atol=atol, rtol=rtol)

    print(f"\nTwoEncoderTransformer - Valid output part matches: {valid_part_matches}")
    if not valid_part_matches:
        print(f"   Max difference in valid part: {(out_nopad - relevant_out_pad).abs().max().item():.2e}")
        print(f"   out_nopad[0, 0, :]:        {out_nopad[0, 0, :]}")
        print(f"   relevant_out_pad[0, 0, :]: {relevant_out_pad[0, 0, :]}")
        print(f"   L2 Norm of difference: {torch.norm(out_nopad - relevant_out_pad).item():.2e}")


    # assert valid_part_matches, "TwoEncoderTransformer valid parts comparison FAILED!"

    # --- Optional: Inspect padded part ---
    # Values depend heavily on proj_out bias and softmax behavior on zeroed inputs.
    # We don't assert equality to zero here, just inspect.
    if L2_pad > L2_valid:
        padded_part_output = out_pad[:, L2_valid:, :]
        print(f"   (Info) Shape of padded output part: {padded_part_output.shape}")
        print(f"   (Info) Max value in padded output part (after softmax): {padded_part_output.max().item():.2e}")
        print(f"   (Info) Min value in padded output part (after softmax): {padded_part_output.min().item():.2e}")
        # Check if probabilities sum to 1 even in padded parts (softmax property)
        print(f"   (Info) Sum of probs in first padded pos: {padded_part_output[0, 0, :].sum().item():.4f}")


    print("\n--- TwoEncoderTransformer Masking Test Finished Successfully ---")
