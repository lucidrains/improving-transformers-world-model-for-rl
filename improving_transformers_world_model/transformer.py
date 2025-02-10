from math import ceil

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import repeat, pack

from vector_quantize_pytorch import VectorQuantize

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def is_empty(t):
    return t.numel() == 0

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
    return block_causal_mask[:seq_len, :seq_len]

# nearest-neighbor tokenizer

class NearestNeighborTokenizer(Module):
    def __init__(
        self,
        dim,
        distance_threshold,
        max_codes = 100_000,
        no_code_id = -1
    ):
        super().__init__()
        self.no_code_id = no_code_id

        self.distance_threshold = distance_threshold

        codes = torch.zeros(max_codes, dim)
        self.register_buffer('_codes', codes) # ran into trouble in the past with dynamically sized buffers, just keep a static shape

        self.register_buffer('num_codes', tensor(0))
        self.register_buffer('num_times_activated', torch.ones(max_codes))

    @property
    def codes(self):
        num_codes = self.num_codes.item()
        return self._codes[:num_codes]

    def add_code_(self, code):
        index = self.num_codes.item()
        self._codes[index].copy_(code)
        self.num_codes.add_(1)

    def add_codes_(self, codes):
        codes, _ = pack_one(codes, '* d')

        codes_added = 0

        # naive approach, adding one code at a time until set of codes all have a neighbor

        while not is_empty(codes):
            first_code, codes = codes[0], codes[1:]

            self.add_code_(first_code)

            is_outside_dist_threshold = ~((torch.cdist(codes, self.codes) ** 2) <= self.distance_threshold).any(dim = -1)
            codes = codes[is_outside_dist_threshold]

            codes_added += 1

        return codes_added

    def forward(
        self,
        x
    ):
        num_codes, no_code_id, device = self.num_codes.item(), self.no_code_id, x.device

        if num_codes == 0:
            self.add_codes_(x)
            return torch.full(x.shape[:-1], no_code_id, device = device)

        # euclidean distance

        distance_sq = torch.cdist(x, self.codes) ** 2

        # within distance threshold set at init

        within_dist_threshold = (distance_sq <= self.distance_threshold).any(dim = -1)

        # if any observations are outside of distance threshold, need to set the new codes

        if self.training and not within_dist_threshold.all():
            new_codes = x[~within_dist_threshold]
            self.add_codes_(new_codes)

        # nearest neighbors by argmin - eq (1) in paper

        nearest_neighbor_ids = distance_sq.argmin(dim = -1)
        nearest_neighbor_ids = torch.where(within_dist_threshold, nearest_neighbor_ids, no_code_id)

        return nearest_neighbor_ids

# transformer

class BlockCausalTransformer(Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
