import threading
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def _encode_samples(self, idxes):
        '''
        :param buffers: {key: array(current_num_episodes_in_buffer x T x dim_key)}
        :param batch_size: batch_size for training
        :return: transitions data (o, o_2, u, r)
        '''
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))

        transitions = {'o': np.array(obses_t), 'o_2': np.array(obses_tp1),
                       'u': np.array(actions), 'r': np.array(rewards)}
        return transitions
        # return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
        '''
        # and finish this function asap.
        T = buffers['u'].shape[1]
        max_episode_num = buffers['u'].shape[0]
        # Select which episodes and time steps to use
        episode_idxs = np.random.randint(0, max_episode_num, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: buffers[key][episode_idxs, t_samples].copy()
                       for key in buffers.keys()}
        return transitions
        '''
    def sample(self, batch_size):
        """
        Note:
            self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
            self.current_size : current number of episodes stored in self.buffers.
        Returns a dict {key: array(batch_size x shapes[key])}
        """
        # buffers = {}
        #
        # with self.lock:
        #     assert self.current_size > 0
        #     for key in self.buffers.keys():
        #         buffers[key] = self.buffers[key][:self.current_size]
        #
        # buffers['o_2'] = buffers['o'][:, 1:, :]
        #
        # transitions = self._encode_samples(buffers, batch_size)
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_samples(idxes)

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        # TODO: Fix the dimensions of episode_batch (np.squeeze type of correction)
        # TODO: Fix this for loop sometime in the future
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        print('episode_batch['u'].shape: {}'.format(episode_batch['u'].shape))
        episode_batch['u'] = np.squeeze(episode_batch['u'])
        episode_batch['o'] = np.squeeze(episode_batch['o'])
        episode_batch['r'] = np.squeeze(episode_batch['r'])
        if len(episode_batch['u'].shape) == 1:
            print('episode only contains one data sample!!')
            return
        episode_len, _ = episode_batch['u'].shape
        print('episode length in store_episode: {}'.format(episode_len))
        for i in range(episode_len-1):
            o = episode_batch['o'][i]; o_2 = episode_batch['o'][i+1]
            u = episode_batch['u'][i]; r = episode_batch['r'][i]
            data = (o, u, r, o_2)
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize
        '''
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T
        '''

    def clear_buffer(self):
        # TODO Fix this
        self._storage = []
        self._next_idx = 0