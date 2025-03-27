from collections import Counter

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.model_based.dreamer.episode_replay_buffer import (
    EpisodeReplayBufferLowLevelRAPS,
)


def test_prioritized_replay():
    """
    Create 10 trajectories ranging from 100 to 1000
    Thus, the probability of each would be 100/C , 200/C, ..., where C=4500
    Which would be similar to the original 1/45, 2/45
    """
    # Dummy environment
    env = NormalizedBoxEnv(HalfCheetahEnv())

    # Create buffer with dummy parameters
    buffer = EpisodeReplayBufferLowLevelRAPS(
        env=env,
        observation_dim=20,
        action_dim=5,
        max_replay_buffer_size=1000,
        max_path_length=100,
        num_low_level_actions_per_primitive=5,
        low_level_action_dim=5,
        replace=True,
        batch_length=50,
        use_batch_length=False,
        prioritize_fraction=1,
        uniform_priorities=False,
    )

    assert buffer is not None

    # create a rewards array with 10 trajectories of 100 steps
    buffer._rewards = np.zeros((10, 100, 1))
    buffer._size = 10
    for i in range(10):
        buffer._rewards[i, -1, -1] = (i + 1) * 100

    iters = 10000
    eps = 1e-4

    index = buffer.random_batch(iters)["rewards"].sum(axis=1) // 100
    index = index.astype(int)
    index = np.squeeze(index)
    counts = Counter(index.astype(int).tolist())

    for index, count in enumerate(counts):
        assert count / iters - (index + 1) / 45 <= eps
