from __future__ import annotations

from math import ceil
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, cdist, cat
from torch.nn import Module, ModuleList, Linear
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from vector_quantize_pytorch import VectorQuantize

from hyper_connections import get_init_and_expand_reduce_stream_functions

from improving_transformers_world_model.distributed import all_gather_variable_dim

from hl_gauss_pytorch import HLGaussLoss

from rotary_embedding_torch import RotaryEmbedding

import einx
from einops import rearrange, repeat, reduce, pack, unpack, einsum
from einops.layers.torch import Rearrange

from improving_transformers_world_model.tensor_typing import (
    Float,
    Int,
    Bool
)

from tqdm import tqdm

# ein notation

# b - batch
# c - channels of game video
# t - time steps
# h - height
# w - width
# n - sequence (flattened spacetime)
# h - attention heads
# a - number of actions
# l - logits / prediction bins

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg

    return None

def divisible_by(num, den):
    return (num % den) == 0

def xnor(x, y):
    return not (x ^ y)

# tensor helpers

def pack_one(t, pattern):
    return pack([t], pattern)

def pack_one_with_inverse(t, pattern, inv_pattern = None):
    packed, ps = pack([t], pattern)

    def inverse(output, override_pattern = None):
        unpacked, = unpack(output, ps, default(override_pattern, inv_pattern, pattern))
        return unpacked

    return packed, inverse

def is_empty(t):
    return t.numel() == 0

def cache_detach_(cache):
    return tree_map(lambda t: t.detach_() if is_tensor(t) else t, cache)

def to_device(tree, device):
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, tree)

def inputs_to_model_device(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        args, kwargs = to_device((args, kwargs), self.device)
        return fn(self, *args, **kwargs)
    return inner

def outputs_to_device(device):
    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            out = fn(*args, **kwargs)
            return to_device(out, device)
        return inner
    return decorator

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

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
        self.max_codes = max_codes
        self.no_code_id = no_code_id

        self.distance_threshold = distance_threshold

        codes = torch.zeros(max_codes, dim)
        self.register_buffer('_codes', codes) # ran into trouble in the past with dynamically sized buffers, just keep a static shape

        self.register_buffer('num_codes', tensor(0))
        self.register_buffer('num_times_activated', torch.ones(max_codes))

    @property
    def is_at_max_codes(self):
        return self.num_codes.item() == self.max_codes

    @property
    def codes(self):
        num_codes = self.num_codes.item()
        return self._codes[:num_codes]

    def add_code_(
        self,
        code: Float['d']
    ):
        index = self.num_codes.item()
        self._codes[index].copy_(code)
        self.num_codes.add_(1)

    def add_codes_(
        self,
        codes: Float['... d']
    ):
        codes, _ = all_gather_variable_dim(codes)

        codes, _ = pack_one(codes, '* d')

        codes_added = 0

        # naive approach, adding one code at a time until set of codes all have a neighbor

        while not is_empty(codes) and not self.is_at_max_codes:
            first_code, codes = codes[0], codes[1:]

            self.add_code_(first_code)

            is_outside_dist_threshold = ~((torch.cdist(codes, self.codes) ** 2) <= self.distance_threshold).any(dim = -1)
            codes = codes[is_outside_dist_threshold]

            codes_added += 1

        return codes_added

    def codes_from_indices(
        self,
        indices: Int['b *dims']
    ) -> Float['b *dims d']:
        return einx.get_at('[c] d, ... -> ... d', self._codes, indices)

    def forward(
        self,
        x: Float['b *dims d'],
        ignore_dist_threshold = None,
        freeze = False

    ) -> Int['b *dims']:

        ignore_dist_threshold = default(ignore_dist_threshold, not self.training or self.is_at_max_codes)

        x, inverse_pack_one = pack_one_with_inverse(x, 'b * d', 'b *')

        num_codes, no_code_id, device = self.num_codes.item(), self.no_code_id, x.device

        if num_codes == 0:
            self.add_codes_(x)
            ids = cdist(x, self.codes).argmin(dim = -1)
            return inverse_pack_one(ids)

        # euclidean distance

        distance_sq = cdist(x, self.codes) ** 2

        # early return with closest code if evaluating, ignoring the distance threshold

        if ignore_dist_threshold:
            ids = distance_sq.argmin(dim = -1)
            return inverse_pack_one(ids)

        # within distance threshold set at init

        within_dist_threshold = (distance_sq <= self.distance_threshold).any(dim = -1)

        # nearest neighbors by argmin - eq (1) in paper

        nearest_neighbor_ids = distance_sq.argmin(dim = -1)
        nearest_neighbor_ids = torch.where(within_dist_threshold, nearest_neighbor_ids, no_code_id)

        # early return if not training

        if freeze or not self.training:
            return inverse_pack_one(nearest_neighbor_ids)

        # if any observations are outside of distance threshold, need to set the new codes

        all_within_dist_threshold = within_dist_threshold.all()

        all_within_dist_threshold, _ = all_gather_variable_dim(all_within_dist_threshold)

        if all_within_dist_threshold.all():
            return inverse_pack_one(nearest_neighbor_ids)

        new_codes = x[~within_dist_threshold]

        self.add_codes_(new_codes)

        new_code_ids = cdist(new_codes, self.codes).argmin(dim = -1)

        nearest_neighbor_ids.masked_fill_(~within_dist_threshold, new_code_ids)

        return inverse_pack_one(nearest_neighbor_ids)

# attention

class BlockCausalAttention(Module):
    def __init__(
        self,
        dim,
        block_size,
        heads = 8,
        dim_head = 64,
        accept_value_residual = False,
        dropout = 0.1
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

        # dropout

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout # for flex attention

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
        flex_attn_block_mask = None,
        cache = None,
    ):
        x = self.norm(x)

        seq_len, device = x.shape[1], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, qkv)

        orig_v = v

        # value residual

        if exists(value_residual):
            value_residual_mix = self.to_value_residual_mix(x)
            v = v.lerp(value_residual, value_residual_mix)

        # handle cache

        is_inferencing = exists(cache)

        if is_inferencing:
            ck, cv = cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        next_cache = (k, v)

        # rotary embed

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # handle a recent advance, value residual

        assert xnor(exists(value_residual), self.accept_value_residual)

        if not is_inferencing and exists(flex_attn_block_mask):

            dropout_p = self.dropout_p if self.training else 0.

            out = flex_attention(
                q, k, v,
                dropout_p = dropout_p,
                block_mask = flex_attn_block_mask
            )
        else:
            # block causal mask

            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if not is_inferencing:
                block_causal_mask = nonflex_block_causal_mask(seq_len, self.block_size, device = device)
                sim = sim.masked_fill(~block_causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads and combine out

        out = self.merge_heads(out)

        return self.to_out(out), (orig_v, next_cache)

# feedforward, swi glu variant from Shazeer et al.

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        expand_factor = 4.,
        dropout = 0.1
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_hidden = int(dim * expand_factor * 2 / 3)
        self.proj_in = Linear(dim, dim_hidden * 2)
        self.proj_out = Linear(dim_hidden, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.gelu(gates)

        x = self.dropout(x)

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
        use_flex_attn = False,
        dropout_attn = 0.1,
        dropout_ff = 0.1
    ):
        super().__init__()
        self.dim = dim

        layers = []

        assert not (use_flex_attn and not exists(flex_attention))
        self.use_flex_attn = use_flex_attn

        self.block_size = block_size

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, disable = num_residual_streams == 1)

        # layers

        for i in range(depth):
            is_first = i == 0

            attn = BlockCausalAttention(dim = dim, dim_head = dim_head, heads = heads, block_size = block_size, accept_value_residual = not is_first, dropout = dropout_attn)
            ff = SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, dropout = dropout_ff)

            layers.append(ModuleList([
                init_hyper_conn(branch = attn),
                init_hyper_conn(branch = ff)
            ]))

        self.layers = ModuleList(layers)

        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        tokens,
        cache = None,
        return_cache = False,
        remove_cache_len_from_time = True
    ):

        seq_len = tokens.shape[1]

        # hyper connection residual streams

        tokens = self.expand_streams(tokens)

        # handle cache

        if exists(cache) and remove_cache_len_from_time:
            cache_len = cache[0][0].shape[-2]
            tokens = tokens[:, cache_len:]

        iter_cache = iter(default(cache, []))
        next_cache_kvs = []

        # value residuals

        first_attn_values = None

        # maybe flex attention

        flex_attn_block_mask = None

        if not exists(cache) and self.use_flex_attn:
            flex_attn_block_mask = create_block_causal_mask(seq_len, self.block_size)

        # layers of attention and feedforward

        for attn, ff in self.layers:
            tokens, (attn_values, attn_cache_kv) = attn(
                tokens,
                cache = next(iter_cache, None),
                value_residual = first_attn_values,
                flex_attn_block_mask = flex_attn_block_mask
            )

            next_cache_kvs.append(attn_cache_kv)
        
            first_attn_values = default(first_attn_values, attn_values)

            tokens = ff(tokens)

        # reduce residual streams

        tokens = self.reduce_streams(tokens)

        embed = self.norm(tokens)

        if not return_cache:
            return embed

        return embed, next_cache_kvs

# world model
# their proposed successful world model is a memorizing nearest neighbor tokenizer + block causal transformer

class WorldModel(Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels,
        num_actions = 0,            # if set to 0, disabled
        reward_min_value = 0.,
        reward_max_value = 1.,
        reward_num_bins = 0.,       # if set to 0, disabled
        transformer_use_token_embed = True,
        tokenizer: NearestNeighborTokenizer | Module | dict = dict(),
        transformer: BlockCausalTransformer | dict = dict(),
        hl_gauss_loss_kwargs: dict = dict(
            sigma_to_bin_ratio = 1. # amount of label smoothing
        ),
        is_terminal_loss_weight = 1.,
        reward_loss_weight = 1.,
        dropout_embed = 0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.channels = channels

        assert divisible_by(image_size, patch_size)

        self.state_to_patches = Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patches_to_state = Rearrange('b t h w (p1 p2 c) -> b c t (h p1) (w p2)', p1 = patch_size, p2 = patch_size)

        patch_dim = (image_size // patch_size)
        patch_size_with_channel = (patch_size ** 2) * channels
        patches_per_image = patch_dim ** 2

        if isinstance(transformer, dict):
            transformer = BlockCausalTransformer(**transformer)

        self.transformer = transformer
        assert transformer.block_size == patches_per_image, f'transformer block size is recommended to be the number of patches per game image, which is {patches_per_image}'

        if isinstance(tokenizer, dict):
            tokenizer = NearestNeighborTokenizer(**tokenizer)

        self.tokenizer = tokenizer

        self.state_token_embed = nn.Embedding(tokenizer.max_codes, transformer.dim) if transformer_use_token_embed else None

        # projecting in and out from patches to model dimensions

        model_dim = transformer.dim
        self.proj_in = nn.Linear(patch_size_with_channel, model_dim)

        self.to_state_pred = nn.Linear(model_dim, tokenizer.max_codes)

        # is terminal state

        self.to_is_terminal_pred = nn.Sequential(
            nn.Linear(model_dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

        # action conditioning related

        self.num_actions = num_actions

        can_cond_on_actions = num_actions > 0
        self.can_cond_on_actions = can_cond_on_actions

        self.action_embed_sos = nn.Parameter(torch.zeros(model_dim))

        self.action_embed = nn.Embedding(num_actions, model_dim) if can_cond_on_actions else None

        # reward related

        can_pred_reward = reward_num_bins > 0

        self.can_pred_reward = can_pred_reward

        self.to_reward_embed = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        ) if can_pred_reward else None

        self.to_reward_pred = nn.Linear(model_dim, reward_num_bins) if can_pred_reward else None

        self.hl_gauss_loss = HLGaussLoss(
            min_value = reward_min_value,
            max_value = reward_max_value,
            num_bins = reward_num_bins,
            **hl_gauss_loss_kwargs
        ) if can_pred_reward else None

        # dropouts

        self.dropout_embed = nn.Dropout(dropout_embed)

        # loss related

        self.is_terminal_loss_weight = is_terminal_loss_weight

        self.reward_loss_weight = reward_loss_weight

        # zero for device and dummy

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @torch.no_grad()
    def sample(
        self,
        prompt: Float['b c t h w'],
        time_steps,
        cache = None,
        rewards: Float['b t'] | None = None,
        actions: Int['b t a'] | None = None,
        filter_fn = min_p_filter,
        filter_kwargs: dict = dict(),
        temperature = 1.5,
        return_token_ids = False,
        return_cache = False,
        return_rewards_and_done = False
    ):
        was_training = self.training

        self.eval()

        batch, prompt_time, device = prompt.shape[0], prompt.shape[-3], prompt.device

        assert prompt_time <= time_steps, f'nothing to sample, as prompt already is greater or equal to desired number of time steps'

        ids = self.forward_tokenizer(prompt)

        if return_rewards_and_done:

            if not exists(rewards):
                rewards = torch.zeros((batch, prompt_time), device = device)

            is_terminals = torch.zeros((batch, prompt_time), device = device, dtype = torch.bool)

        for _ in tqdm(range(time_steps - prompt_time)):

            (state_logits, reward_logits, is_terminal_logits), cache = self.forward(
                ids,
                actions = actions,
                rewards = rewards,
                cache = cache,
                return_loss = False,
                return_cache = True
            )

            # sample next state

            state_logits = state_logits[:, -1:] # last timestep logits

            state_logits = filter_fn(state_logits, **filter_kwargs)
            next_state_ids = gumbel_sample(state_logits, temperature = temperature, dim = -1, keepdim = False)

            ids = cat((ids, next_state_ids), dim = 1)

            # next reward

            if exists(rewards):
                next_rewards = self.hl_gauss_loss(reward_logits[:, -1:])

                rewards = cat((rewards, next_rewards), dim = 1)

            # done state

            if return_rewards_and_done:
                is_terminal = is_terminal_logits.sigmoid().round().bool()

                is_terminals = cat((is_terminals, is_terminal), dim = -1)

            # will sticky the action to the very last action for now

            if exists(actions):
                action = actions[:, -1:]

                actions = cat((actions, action), dim = 1)

        self.train(was_training)

        out = ids

        if return_rewards_and_done:
            out = (out, rewards, is_terminals)

        if return_token_ids:

            if not return_cache:
                return out

            return out, cache

        nearest_neighbor_codes = self.tokenizer.codes_from_indices(ids)

        state = self.patches_to_state(nearest_neighbor_codes)

        if return_rewards_and_done:
            out = (state, rewards, is_terminals)

        if not return_cache:
            return out

        return out, cache

    def forward_tokenizer(
        self,
        state: Float['b c t h w']
    ) -> Int['b n d']:

        patches = self.state_to_patches(state)
        return self.tokenizer(patches)

    @inputs_to_model_device
    def forward(
        self,
        state_or_token_ids: Float['b c t h w'] | Int['b t h w'],
        rewards: Float['b t'] | None = None,
        actions: Int['b t a'] | None = None, # values of < 0 as padding, allowing for multiple actions to be summed per timestep
        is_terminal: Bool['b t'] | None = None, # learn to predict the terminal state, for the agent interacting with the world model in MDP manner
        cache = None,
        remove_cache_len_from_time = True,
        return_cache = False,
        detach_cache = False,
        return_loss = True,
        return_loss_breakdown = False,
        freeze_tokenizer = True
    ):
        batch = state_or_token_ids.shape[0]

        assert xnor(exists(rewards), self.can_pred_reward)
        assert xnor(exists(actions), self.can_cond_on_actions)

        if state_or_token_ids.dtype  == torch.float:
            state = state_or_token_ids
            token_ids = self.forward_tokenizer(state)
        else:
            token_ids = state_or_token_ids

        if return_loss:
            token_ids, state_labels = token_ids[:, :-1], token_ids[:, 1:]

            is_terminal_labels = is_terminal[:, 1:]

            actions, last_action = actions[:, :-1], actions[:, -1:]
            rewards, last_reward = rewards[:, :-1], rewards[:, -1:]

        # either use own learned token embeddings
        # or project the codes (which are just the nearest neighbor memorized patch) and project
        # todo: maybe allow for a bit of both with learned mix

        if exists(self.state_token_embed):
            tokens = self.state_token_embed(token_ids)
        else:
            tokens = self.tokenizer.codes_from_indices(token_ids)
            tokens = self.proj_in(tokens)

        # state embed dropout

        tokens = self.dropout_embed(tokens)

        # maybe reward conditioning

        if exists(rewards):
            reward_embeds = self.to_reward_embed(rewards)

            tokens = einx.add('b t h w d, b t d -> b t h w d', tokens, reward_embeds)

        # maybe action conditioning

        if exists(actions):
            no_actions = actions < 0
            actions = actions.masked_fill(no_actions, 0)
            action_embeds = self.action_embed(actions)

            action_embeds = einx.where('b t n, b t n d, -> b t n d', ~no_actions, action_embeds, 0.)
            action_embeds = reduce(action_embeds, 'b t n d -> b t d', 'sum')

            action_embed_sos = repeat(self.action_embed_sos, 'd -> b 1 d', b = batch)
            action_embeds = cat((action_embed_sos, action_embeds[:, :-1]), dim = 1)

            tokens = einx.add('b t h w d, b t d -> b t h w d', tokens, action_embeds)

        is_inferencing = exists(cache)

        # pack the spacetime dimension into one sequence for block causal attention

        tokens, inverse_space = pack_one_with_inverse(tokens, 'b t * d')

        flattened_space_dim = tokens.shape[-2]

        def inverse_time(t):
            return rearrange(t, 'b (t n) d -> b t n d', n = flattened_space_dim)

        tokens = rearrange(tokens, 'b t n d-> b (t n) d')

        embeds, next_cache = self.transformer(tokens, remove_cache_len_from_time = remove_cache_len_from_time, cache = cache, return_cache = True)

        if detach_cache:
            cache_detach_(next_cache)

        state_logits = self.to_state_pred(embeds)

        state_logits = inverse_time(state_logits)
        state_logits = inverse_space(state_logits)

        # maybe pool embeds across space if predicting reward and terminal

        if not is_inferencing:
            embeds_with_spacetime = inverse_space(inverse_time(embeds))

            # average pool for embeddings to project into rewards per time step

            embeds_with_time = reduce(embeds_with_spacetime, 'b t ... d -> b t d', 'mean')
        else:
            embeds_with_space = inverse_space(embeds, 'b * d')

            embeds_with_time = reduce(embeds_with_space, 'b ... d -> b 1 d', 'mean')

        # reward and terminal

        reward_logits = None

        if self.can_pred_reward:
            reward_logits = self.to_reward_pred(embeds_with_time)

        is_terminal_logits = self.to_is_terminal_pred(embeds_with_time)

        if not return_loss:
            logits = (state_logits, reward_logits, is_terminal_logits)

            if not return_cache:
                return logits

            return logits, next_cache

        state_loss = F.cross_entropy(
            rearrange(state_logits, 'b ... l -> b l (...)'),
            rearrange(state_labels, 'b ... -> b (...)'),
            ignore_index = -1
        )

        # maybe predict reward

        reward_loss = self.zero

        if exists(rewards):
            reward_loss = self.hl_gauss_loss(reward_logits, rewards)

        # maybe predict if state at given time is terminal

        is_terminal_loss = self.zero

        if exists(is_terminal):
            is_terminal_loss = F.binary_cross_entropy_with_logits(
                is_terminal_logits,
                is_terminal_labels.float()
            )

        # add all losses

        total_loss = (
            state_loss +
            (is_terminal_loss * self.is_terminal_loss_weight) +
            (reward_loss * self.reward_loss_weight)
        )

        breakdown = (state_loss, reward_loss, is_terminal_loss)

        if not return_loss_breakdown and not return_cache:
            return total_loss

        if return_loss_breakdown and not return_cache:
            return total_loss, breakdown

        return total_loss, (breakdown, next_cache)
