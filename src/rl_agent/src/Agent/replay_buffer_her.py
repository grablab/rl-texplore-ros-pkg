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

    def sample_her(self, batch_size):
        episode_idxs = np.random.randint(0, )
        episode_len_arr =


    def sample_a2c(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_samples_a2c(idxes)

    def store_episode_a2c(self, episode_batch):
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        print('episode_batch['u'].shape: {}'.format(episode_batch['u'].shape))
        episode_batch['u'] = np.squeeze(episode_batch['u'])
        episode_batch['o'] = np.squeeze(episode_batch['o'])
        episode_batch['r'] = np.squeeze(episode_batch['r'])
        episode_batch['v'] = np.squeeze(episode_batch['v'])
        episode_batch['g'] = np.squeeze(episode_batch['g'])
        if len(episode_batch['u'].shape) == 1:
            print('episode only contains one data sample!!')
            return
        episode_len, _ = episode_batch['u'].shape
        print('episode length in store_episode: {}'.format(episode_len))
        for i in range(episode_len-1):
            o = episode_batch['o'][i]; o_2 = episode_batch['o'][i+1]
            u = episode_batch['u'][i]; r = episode_batch['r'][i]
            v = episode_batch['v'][i]
            data = (o, u, r, o_2, v)
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize