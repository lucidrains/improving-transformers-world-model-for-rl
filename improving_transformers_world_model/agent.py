from __future__ import annotations

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops.layers.torch import Reduce

from improving_transformers_world_model.tensor_typing import (
    Float,
    Int,
    Bool
)

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
            Reduce('b c h w -> b c'),
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
        init_conv_kernel = 7
    ):
        super().__init__()
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

        self.to_value_pred = nn.Sequential(
            Reduce('b c h w -> b c'),
            nn.Linear(dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(
        self,
        state: Float['b c h w'],
        returns: Float['b'] | None = None

    ) -> Float['b'] | Float['']:

        embed = self.proj_in(state)

        for layer in self.layers:
            embed = layer(embed) + embed

        values = self.to_value_pred(embed)

        if not exists(returns):
            return values

        return F.mse_loss(values, returns)

class Agent(Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic