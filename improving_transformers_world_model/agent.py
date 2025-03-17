from __future__ import annotations
from typing import NamedTuple

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
from torch.distributions import Categorical

import torch.nn.functional as F

from einops.layers.torch import Reduce

from improving_transformers_world_model.world_model import (
    WorldModel
)

from improving_transformers_world_model.tensor_typing import (
    Float,
    Int,
    Bool
)

from hl_gauss_pytorch import HLGaussLayer

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def calc_entropy(prob, eps = 1e-20, dim = -1):
    return -(prob * log(prob, eps)).sum(dim = dim)

# classes

class Actor(Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        channels,
        num_actions,
        num_layers = 3,
        expansion_factor = 2.,
        init_conv_kernel = 7
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        dim_hidden = int(expansion_factor * dim)

        self.proj_in = nn.Conv2d(channels, dim, init_conv_kernel, stride = 2, padding = init_conv_kernel // 2)

        layers = []

        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(dim, dim_hidden, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(dim_hidden, dim, 3, padding = 1),
            )

            layers.append(layer)

        self.layers = ModuleList(layers)

        self.to_actions_pred = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, num_actions),
        )

    def forward(
        self,
        state: Float['b c h w']
    ) -> Float['b a']:

        embed = self.proj_in(state)

        for layer in self.layers:
            embed = layer(embed) + embed

        action_logits = self.to_actions_pred(embed)

        return action_logits

class Critic(Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        channels,
        num_layers = 4,
        expansion_factor = 2.,
        init_conv_kernel = 7,
        use_regression = False,
        hl_gauss_loss_kwargs = dict(
            min_value = 0.,
            max_value = 5.,
            num_bins = 32,
            sigma = 0.5,
        )
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        dim_hidden = int(expansion_factor * dim)

        self.proj_in = nn.Conv2d(channels, dim, init_conv_kernel, stride = 2, padding = init_conv_kernel // 2)

        layers = []

        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(dim, dim_hidden, 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(dim_hidden, dim, 3, padding = 1),
            )

            layers.append(layer)

        self.layers = ModuleList(layers)

        self.pool = Reduce('b c h w -> b c', 'mean')

        self.to_value_pred = HLGaussLayer(
            dim = dim,
            hl_gauss_loss = hl_gauss_loss_kwargs
        )

    def forward(
        self,
        state: Float['b c h w'],
        returns: Float['b'] | None = None

    ) -> Float['b'] | Float['']:

        embed = self.proj_in(state)

        for layer in self.layers:
            embed = layer(embed) + embed

        pooled = self.pool(embed)
        values = self.to_value_pred(pooled)

        if not exists(returns):
            return values

        return F.mse_loss(values, returns)

# memory

FrameState = Float['c h w']
Scalar = Float['']
Loss = Scalar

class Memory(NamedTuple):
    state:           FrameState
    action:          Int['a']
    action_log_prob: Scalar
    reward:          Scalar
    value:           Scalar
    done:            Bool['']

Memories = list[Memory]

# actor critic agent

class Agent(Module):
    def __init__(
        self,
        actor: Actor | dict,
        critic: Critic | dict,
        actor_eps_clip = 0.2, # clipping
        actor_beta_s = .01,   # entropy weight
    ):
        super().__init__()

        if isinstance(actor, dict):
            actor = Actor(**actor)

        if isinstance(critic, dict):
            critic = Critic(**critic)

        self.actor = actor
        self.critic = critic

        self.actor_eps_clip = actor_eps_clip
        self.actor_beta_s = actor_beta_s

        assert actor.image_size == critic.image_size and actor.channels == critic.channels

    def policy_loss(
        self,
        state: Float['b c h w'],
        actions: Int['b'],
        old_log_probs: Float['b'],
        values: Float['b'],
        returns: Float['b'],
    ) -> Loss:

        batch = values.shape[0]
        advantages = F.layer_norm(returns - values, (batch,))

        action_logits = self.actor(state)
        prob = action_logits.softmax(dim = -1)

        distrib = Categorical(prob)
        log_probs = distrib.log_prob(actions)

        ratios = (log_probs - old_log_probs).exp()

        # ppo clipped surrogate objective

        clip = self.actor_eps_clip

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1. - clip, 1. + clip) * advantages

        action_entropy = calc_entropy(prob) # encourage exploration
        policy_loss = torch.min(surr1, surr2) - self.actor_beta_s * action_entropy

        return policy_loss

    def critic_loss(
        self,
        state: Float['b c h w'],
        returns: Float['b']
    ) -> Loss:

        critic_loss = self.critic(state, returns)
        return critic_loss

    def learn(
        self,
        memories: Memories

    ) -> tuple[Loss, ...]:

        raise NotImplementedError

    def forward(
        self,
        world_model: WorldModel,
        init_state: FrameState

    ) -> tuple[
        Memories,
        FrameState
    ]:

        assert world_model.image_size == self.actor.image_size and world_model.channels == self.actor.channels

        raise NotImplementedError
