from __future__ import annotations

import torch
from torch import tensor
from torch.nn import Module

from improving_transformers_world_model.tensor_typing import (
    Float,
    Int,
    Bool
)

# constants

FrameState = Float['c h w']
Scalar = Float['']

# mock env

class Env(Module):
    def __init__(
        self,
        state_shape: tuple[int, ...]
    ):
        super().__init__()
        self.state_shape = state_shape
        self.register_buffer('dummy', tensor(0))

    @property
    def device(self):
        return self.dummy.device

    def reset(
        self
    ) -> FrameState:
        return torch.randn(self.state_space, device = self.device)

    def forward(
        self,
        actions: Int['a'],
    ) -> tuple[
        FrameState,
        Scalar,
        Bool[''],
    ]:
        state = torch.randn(self.state_shape, device = self.device)
        reward = torch.randn(1, device = self.device)
        done = torch.zeros(1, device = self.device)

        return state, reward, done
