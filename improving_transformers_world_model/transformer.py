from math import ceil

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import repeat, pack, einsum
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from hyper_connections import get_init_and_expand_reduce_stream_functions

from improving_transformers_world_model.distributed import all_gather_variable_dim

from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return not (x ^ y)

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

# patch nearest-neighbor tokenizer proposed in section 3.5

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

        # nearest neighbors by argmin - eq (1) in paper

        nearest_neighbor_ids = distance_sq.argmin(dim = -1)
        nearest_neighbor_ids = torch.where(within_dist_threshold, nearest_neighbor_ids, no_code_id)

        # early return if not training

        if not self.training:
            return nearest_neighbor_ids

        # if any observations are outside of distance threshold, need to set the new codes

        all_within_dist_threshold = within_dist_threshold.all()

        all_within_dist_threshold, _ = all_gather_variable_dim(all_within_dist_threshold)

        if not all_within_dist_threshold.all():
            return nearest_neighbor_ids

        new_codes = x[~within_dist_threshold]

        new_codes, _ = all_gather_variable_dim(new_codes)

        self.add_codes_(new_codes)

        return nearest_neighbor_ids

# attention

class BlockCausalAttention(Module):
    def __init__(
        self,
        dim,
        block_size,
        heads = 8,
        dim_head = 64,
        accept_value_residual = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads

        self.block_size = block_size

        self.to_qkv = Linear(dim, dim_inner * 3, bias = False)
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = Linear(dim_inner, dim, bias = False)

        # rope

        self.rotary_emb = RotaryEmbedding(dim_head)

        # value residual learning

        self.accept_value_residual = accept_value_residual

        self.to_value_residual_mix = nn.Sequential(
            Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

    def forward(
        self,
        x,
        value_residual = None,
        flex_attn_block_mask = None
    ):
        x = self.norm(x)

        seq_len, device = x.shape[1], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, qkv)

        orig_v = v

        # rotary embed

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # handle a recent advance, value residual

        assert xnor(exists(value_residual), self.accept_value_residual)

        if exists(value_residual):
            value_residual_mix = self.to_value_residual_mix(x)
            v = v.lerp(value_residual, value_residual_mix)

        if exists(flex_attn_block_mask):
            out = flex_attention(q, k, v, block_mask = flex_attn_block_mask)
        else:
            # block causal mask

            block_causal_mask = nonflex_block_causal_mask(seq_len, self.block_size, device = device)

            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = sim.masked_fill(~block_causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads and combine out

        out = self.merge_heads(out)

        return self.to_out(out), orig_v

# feedforward, swi glu variant from Shazeer et al.

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_hidden = int(dim * expand_factor * 2 / 3)
        self.proj_in = Linear(dim, dim_hidden * 2)
        self.proj_out = Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.gelu(gates)

        return self.proj_out(x)

# transformer

class BlockCausalTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        block_size,
        dim_head = 64,
        heads = 8,
        ff_expand_factor = 4.,
        num_residual_streams = 4,
        use_flex_attn = False
    ):
        super().__init__()
        layers = []

        assert not (use_flex_attn and not exists(flex_attention))
        self.use_flex_attn = use_flex_attn

        self.block_size = block_size

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, disable = num_residual_streams == 1)

        # layers

        for i in range(depth):
            is_first = i == 0

            attn = BlockCausalAttention(dim = dim, dim_head = dim_head, heads = heads, block_size = block_size, accept_value_residual = not is_first)
            ff = SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor)

            layers.append(ModuleList([
                init_hyper_conn(branch = attn),
                init_hyper_conn(branch = ff)
            ]))

        self.layers = ModuleList(layers)

    def sample(self):
        raise NotImplementedError

    def forward(
        self,
        x
    ):
        seq_len = x.shape[1]

        # hyper connection residual streams

        x = self.expand_streams(x)

        # value residuals

        first_attn_values = None

        # maybe flex attention

        flex_attn_block_mask = None

        if self.use_flex_attn:
            flex_attn_block_mask = create_block_causal_mask(seq_len, self.block_size)

        # layers of attention and feedforward

        for attn, ff in self.layers:
            x, attn_values = attn(x, value_residual = first_attn_values, flex_attn_block_mask = flex_attn_block_mask)

            first_attn_values = default(first_attn_values, attn_values)

            x = ff(x)

        # reduce residual streams

        x = self.reduce_streams(x)

        return x
