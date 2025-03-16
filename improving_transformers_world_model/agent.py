import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

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

    def forward(
        self,
        state: Float['b c h w']
    ) -> Float['b na']:

        return state

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

    def forward(
        self,
        state: Float['b c h w']
    ) -> Float['b']:

        return state

class Agent(Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic