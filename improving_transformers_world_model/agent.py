from __future__ import annotations
from typing import NamedTuple

import torch
from torch import nn, cat, stack, tensor, Tensor
from torch.nn import Module, ModuleList

import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

from einops import rearrange
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

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def get_log_prob(logits, indices):
    log_probs = logits.log_softmax(dim = -1)
    indices = rearrange(indices, '... -> ... 1')
    sel_log_probs = log_probs.gather(-1, indices)
    return rearrange(sel_log_probs, '... 1 -> ...')

def calc_entropy(prob, eps = 1e-20, dim = -1):
    return -(prob * log(prob, eps)).sum(dim = dim)

# generalized advantage estimate

def calc_gae(
    rewards: Float['n'],
    values: Float['n+1'],
    masks: Bool['n'],
    gamma = 0.99,
    lam = 0.95
) -> Float['n']:

    device = rewards.device

    gae = 0.
    returns = torch.empty_like(rewards)

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        returns[i] = gae + values[i]

    return returns

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
        self.num_actions = num_actions

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
        state: Float['b c h w'],
        sample_action = False
    ) -> (
        Float['b'] |
        tuple[Int['b'], Float['b']]
    ):

        embed = self.proj_in(state)

        for layer in self.layers:
            embed = layer(embed) + embed

        action_logits = self.to_actions_pred(embed)

        if not sample_action:
            return action_logits

        actions = gumbel_sample(action_logits, dim = -1)

        log_probs = get_log_prob(action_logits, actions)

        return (actions, log_probs)

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
    is_dream:        Bool['']

class MemoriesWithNextState(NamedTuple):
    memories: list[Memory]
    next_state: FrameState

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

        log_probs = get_log_prob(action_logits, actions)

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

    @torch.no_grad()
    def learn(
        self,
        memories: MemoriesWithNextState | list[MemoriesWithNextState],
        lam = 0.95,
        gamma = 0.99,
        batch_size = 16

    ) -> tuple[Loss, ...]:

        if isinstance(memories, MemoriesWithNextState):
            memories = [memories]

        datasets = []

        for one_memories, next_state in memories:

            next_state = rearrange(next_state, 'c 1 h w -> 1 c h w')

            next_value = self.critic(next_state)

            next_value = rearrange(next_value, '1 ... -> ...')

            (
                states,
                actions,
                action_log_probs,
                rewards,
                values,
                dones,
                _
            ) = map(stack, zip(*one_memories))

            values_with_next = cat((values, rearrange(next_value, '... -> 1 ...')), dim = 0)

            # generalized advantage estimate

            returns = calc_gae(rewards, values_with_next, dones, lam = lam, gamma = gamma)

            # memories dataset for updating actor and critic learning

            dataset = TensorDataset(states, actions, action_log_probs, returns, values, dones)

            datasets.append(dataset)

        datasets = ConcatDataset(datasets)

        dataloader = DataLoader(datasets, batch_size = batch_size, shuffle = True)

        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        world_model: WorldModel,
        init_state: FrameState,
        memories: Memories | None = None,
        max_steps = float('inf')

    ) -> MemoriesWithNextState:

        device = init_state.device

        assert world_model.image_size == self.actor.image_size and world_model.channels == self.actor.channels
        assert world_model.num_actions == self.actor.num_actions

        memories = default(memories, [])

        next_state = rearrange(init_state, 'c h w -> 1 c h w')

        # prepare for looping with world model
        # gathering up all the memories of states, actions, rewards for training

        actions = torch.empty((1, 0, 1), device = device, dtype = torch.long)
        action_log_probs = torch.empty((1, 0), device = device, dtype = torch.float32)

        states = rearrange(next_state, '1 c h w -> 1 c 1 h w')

        rewards = torch.zeros((1, 1), device = device, dtype = torch.float32)
        dones = tensor([[False]], device = device)

        last_done = dones[0, -1]
        time_step = states.shape[2] + 1

        world_model_cache = None

        while time_step < max_steps and not last_done:

            action, action_log_prob = self.actor(next_state, sample_action = True)

            action_log_prob = rearrange(action_log_prob, 'b -> b 1')
            action = rearrange(action, 'b -> b 1 1')

            actions = cat((actions, action), dim = 1)
            action_log_probs = cat((action_log_probs, action_log_prob), dim = 1)

            (states, rewards, dones), world_model_cache = world_model.sample(
                prompt = states,
                actions = actions,
                rewards = rewards,
                time_steps = time_step,
                return_rewards_and_done = True,
                return_cache = True,
                cache = world_model_cache
            )

            time_step += 1
            last_done = dones[0, -1]

        # calculate value from critic all at once before storing to memory

        values = self.critic(rearrange(states, '1 c t h w -> t c h w'))
        values = rearrange(values, 't -> 1 t')

        # move all intermediates to cpu and detach and store into memory for learning actor and critic

        states, actions, action_log_probs, rewards, values, dones = tuple(rearrange(t, '1 ... -> ...').cpu() for t in (states, actions, action_log_probs, rewards, values, dones))

        states, next_state = states[:, :-1], states[:, -1:]

        rewards = rewards[:-1]
        values = values[:-1]
        dones = dones[:-1]
        is_dream = tensor(True)

        episode_memories = tuple(Memory(*timestep_tensors) for timestep_tensors in zip(
            rearrange(states, 'c t h w -> t c h w'),
            actions,
            action_log_probs,
            rewards,
            values,
            dones,
            is_dream
        ))

        memories.extend(episode_memories)

        return MemoriesWithNextState(memories, next_state)
