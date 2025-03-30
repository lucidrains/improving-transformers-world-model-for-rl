import torch

from improving_transformers_world_model import (
    WorldModel,
    Agent,
    Impala
)

from improving_transformers_world_model.mock_env import Env

def test_agent():

    # world model

    world_model = WorldModel(
        image_size = 63,
        patch_size = 7,
        channels = 3,
        reward_num_bins = 10,
        num_actions = 5,
        transformer = dict(
            dim = 32,
            depth = 1,
            block_size = 81
        ),
        tokenizer = dict(
            dim = 7 * 7 * 3,
            distance_threshold = 0.5
        )
    )

    state = torch.randn(2, 3, 20, 63, 63)
    rewards = torch.randint(0, 10, (2, 20)).float()
    actions = torch.randint(0, 5, (2, 20, 1))
    is_terminal = torch.randint(0, 2, (2, 20)).bool()

    loss = world_model(state, actions = actions, rewards = rewards, is_terminal = is_terminal)
    loss.backward()

    # agent

    agent = Agent(
        impala = dict(
            image_size = 63,
            channels = 3
        ),
        actor = dict(
            dim = 32,
            num_actions = 5,
        ),
        critic = dict(
            dim = 64,
        )
    )

    env = Env((3, 63, 63))

    dream_memories = agent(world_model, state[0, :, 0], max_steps = 5)

    real_memories = agent.interact_with_env(env, max_steps = 5)

    agent.learn([dream_memories, real_memories])

# burn-in

def world_model_burn_in():
    world_model = WorldModel(
        image_size = 63,
        patch_size = 7,
        channels = 3,
        reward_num_bins = 10,
        num_actions = 5,
        transformer = dict(
            dim = 32,
            depth = 1,
            block_size = 81
        ),
        tokenizer = dict(
            dim = 7 * 7 * 3,
            distance_threshold = 0.5
        )
    )

    state = torch.randn(2, 3, 20, 63, 63) # batch, channels, time, height, width - craftax is 3 channels 63x63, and they used rollout of 20 frames. block size is presumably each image
    rewards = torch.randint(0, 10, (2, 20)).float()
    actions = torch.randint(0, 5, (2, 20, 1))
    is_terminal = torch.randint(0, 2, (2, 20)).bool()

    _, burn_in_cache = world_model(
        state,
        actions = actions,
        rewards = rewards,
        is_terminal = is_terminal,
        return_cache = True,
        return_loss = False,
        detach_cache = True,
    )

    loss = world_model(
        state,
        actions = actions,
        rewards = rewards,
        is_terminal = is_terminal,
        return_loss = True,
        cache = burn_in_cache,
        remove_cache_len_from_time = False
    )

    loss.backward()

def impala_burn_in():
    impala = Impala()
    images = torch.randn(5, 3, 63, 63)

    _, gru_hidden = impala(images)
    cnn_out_rnn_out, gru_hidden = impala(images, gru_hidden = gru_hidden.detach())
