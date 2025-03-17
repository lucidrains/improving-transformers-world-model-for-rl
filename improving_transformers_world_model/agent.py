from __future__ import annotations
from typing import NamedTuple

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
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

Scalar = Float['']

class Memory(NamedTuple):
    state:           Float['c h w']
    action:          Int['a']
    action_log_prob: Scalar
    reward:          Scalar
    value:           Scalar
    done:            Bool['']

# actor critic agent

class Agent(Module):
    def __init__(
        self,
        actor: Actor | dict,
        critic: Critic | dict
    ):
        super().__init__()

        if isinstance(actor, dict):
            actor = Actor(**actor)

        if isinstance(critic, dict):
            critic = Critic(**critic)

        self.actor = actor
        self.critic = critic

        assert actor.image_size == critic.image_size and actor.channels == critic.channels

    def learn(
        self,
        memories: list[Memory]
    ) -> tuple[Scalar, ...]:

        raise NotImplementedError

    def forward(
        self,
        world_model: WorldModel

    ) -> tuple[
        list[Memory],
        Float['c h w']
    ]:

        assert world_model.image_size == self.actor.image_size and world_model.channels == self.actor.channels

        raise NotImplementedError
