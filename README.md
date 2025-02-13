<img src="./fig1.png" width="450px"/>

<img src="./fig2.png" width="450px"/>

## Improving Transformers World Model - Pytorch (wip)

Implementation of the new SOTA for model based RL, from the paper [Improving Transformer World Models for Data-Efficient RL](https://arxiv.org/abs/2502.01591), in Pytorch.

They significantly outperformed DreamerV3 (as well as human experts) with a transformer world model and a less complicated setup, on Craftax (simplified Minecraft environment)

## Install

```bash
$ pip install improving-transformers-world-model
```

## Usage

```python
import torch

from improving_transformers_world_model import (
    WorldModel
)

world_model = WorldModel(
    image_size = 63,
    patch_size = 7,
    channels = 3,
    transformer = dict(
        dim = 512,
        depth = 4,
        block_size = 81
    ),
    tokenizer = dict(
        dim = 7 * 7 * 3,
        distance_threshold = 0.5
    )
)

state = torch.randn(2, 3, 20, 63, 63) # batch, channels, time, height, width - craftax is 3 channels 63x63, and they used rollout of 20 frames. block size is presumably each image

loss = world_model(state)
loss.backward()

# dream up a trajectory to be mixed with real for training PPO

prompts = state[:, :, :2] # prompt frames

imagined_trajectories = world_model.sample(prompts, time_steps = 20)

assert imagined_trajectories.shape == state.shape

```

## Citations

```bibtex
@inproceedings{Dedieu2025ImprovingTW,
    title   = {Improving Transformer World Models for Data-Efficient RL},
    author  = {Antoine Dedieu and Joseph Ortiz and Xinghua Lou and Carter Wendelken and Wolfgang Lehrach and J. Swaroop Guntupalli and Miguel L{\'a}zaro-Gredilla and Kevin Patrick Murphy},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276107865}
}
```
