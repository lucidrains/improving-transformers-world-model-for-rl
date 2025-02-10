from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import repeat

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# for the block teacher forcing proposed in section 3.6

def create_block_causal_mask(seq_len, block_size):

    def create_mask(_, __, q_idx, kv_idx):
        return (q_idx // block_size) >= (kv_idx // block_size)

    block_mask = create_block_mask(create_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

def nonflex_block_causal_mask(seq_len, block_size, device = None):
    blocks = ceil(seq_len / block_size)

    causal_mask = torch.ones((blocks, blocks), device = device, dtype = torch.bool).tril()
    block_causal_mask = repeat(causal_mask, 'i j -> (i bsz1) (j bsz2)', bsz1 = block_size, bsz2 = block_size)
    return block_causal_mask

# transformer

class BlockCausalTransformer(Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
