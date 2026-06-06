from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import h5py
import d4rl
from tqdm import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            online_finetune=False,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.env_name = self.env.spec.id
        self.online_finetune = online_finetune

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


    def load_hdf5(self):

        dataset = d4rl.qlearning_dataset(self.env)

        if not self.online_finetune:
            self._observations = dataset['observations']
            self._next_obs = dataset['next_observations']
            self._actions = dataset['actions']
            if 'antmaze' in self.env_name:
                # self._rewards = (np.expand_dims(np.squeeze(dataset['rewards']), 1) - 0.5) * 4.0
                self._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1) - 1.
            else:
                self._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
            self._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
            self._size = dataset['terminals'].shape[0]
            print ('Number of terminals on: ', self._terminals.sum())
            self._top = self._size
            self._offline_size = self._size
            print('Total samples number: {}'.format(self._size))
        else:
            self._size = dataset['terminals'].shape[0]
            print ('Number of terminals on: ', self._terminals.sum())
            self._top = self._size
            self._offline_size = self._size
            print('Total samples number: {}'.format(self._size))

            self._observations[:self._offline_size] = dataset['observations']
            self._next_obs[:self._offline_size] = dataset['next_observations']
            self._actions[:self._offline_size] = dataset['actions']

            if 'antmaze' in self.env_name:
                # self._rewards[:self._offline_size] = (np.expand_dims(np.squeeze(dataset['rewards']), 1) - 0.5) * 4.0
                self._rewards[:self._offline_size] = np.expand_dims(np.squeeze(dataset['rewards']), 1) - 1.
            else:
                self._rewards[:self._offline_size] = np.expand_dims(np.squeeze(dataset['rewards']), 1)

            self._terminals[:self._offline_size] = np.expand_dims(np.squeeze(dataset['terminals']), 1)


    def normalize_states(self, eps=1e-3):
        self.mean = self._observations.mean(axis=0, keepdims=True)
        self.std = self._observations.std(axis=0, keepdims=True) + eps
        self._observations = (self._observations - self.mean) / self.std
        self._next_obs = (self._next_obs - self.mean) / self.std
        print('=============  finishing state normalization  =================')

    def reward_scale_by_traj_returns(self):
        assert self._offline_size > 0, 'please load offline dataset for reward scale'
        returns, lengths = [], []
        ep_ret, ep_len = 0., 0
        for r, d in zip(self._rewards[:self._offline_size].squeeze(), self._terminals[:self._offline_size].squeeze()):
            ep_ret += float(r)
            ep_len += 1
            if d or ep_len == self.env._max_episode_steps - 1:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0., 0
        lengths.append(ep_len)
        assert sum(lengths) == self._offline_size, 'miscount number of offline data'
        reward_scale = self.env._max_episode_steps / (max(returns) - min(returns) + 1E-8)
        print('========= reward scale by traj returns: {}'.format(reward_scale))
        return reward_scale

