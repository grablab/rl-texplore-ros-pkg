import threading

import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T):
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        # self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        # Below: only works for python3.x
        #self.buffers = {key: np.empty([self.size, *shape])
        #                for key, shape in buffer_shapes.items()}
        self.buffers = {}
        for key, shape in buffer_shapes.items():
            expanded_shape = [self.size] + list(shape)
            self.buffers[key] = np.empty(expanded_shape)

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def _encode_samples(self, buffers, batch_size):
        '''
        :param buffers: {key: array(current_num_episodes_in_buffer x T x dim_key)}
        :param batch_size: batch_size for training
        :return: transitions data (o, o_2, u, r)
        '''
        #TODo: Read _sample_her_transitions(episode_batch, batch_size_in_transitions)
        # and finish this function asap.
        T = buffers['u'].shape[1]
        max_episode_num = buffers['u'].shape[0]
        # Select which episodes and time steps to use
        episode_idxs = np.random.randint(0, max_episode_num, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: buffers[key][episode_idxs, t_samples].copy()
                       for key in buffers.keys()}
        return transitions

    def sample(self, batch_size):
        """
        Note:
            self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
            self.current_size : current number of episodes stored in self.buffers.
        Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]

        transitions = self._encode_samples(buffers, batch_size)
        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx