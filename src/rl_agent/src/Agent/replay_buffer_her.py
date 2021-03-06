import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size, num_rollouts, sample_transitions):
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self._storage = []
        # self.buffers = dict(o=[], u=[], r=[], d=[], mu=[])
        self.num_rollouts = num_rollouts  # nsteps + 1
        # -> This self.nsteps should be num_rollouts. self.nsteps is the number of steps in a sampled trajectory, which should be determined by a model definition in acer.py
        self.size = int(10E6)
        self.buffer_shapes = {'o': 13, 'u': 1, 'r': 1, 'done': 1, 'mu': 9, 'ag': 13,
                              'drop': 1, 'g':13}
        self.buffers = {key: np.empty([self.size, self.num_rollouts, shape])
                        for key, shape in self.buffer_shapes.items()}
        self.buffers['drop_time_steps'] = np.empty([self.size, 1])
        # [num_of_episodes, nsteps, 13] if 'o'
        self.sample_transitions = sample_transitions
        self._maxsize = size
        self._next_idx = 0
        self.num_in_buffer = 0  # I think this is from the new replay buffer code from OpenAI
        self.current_size = 0  # counts the number of episodes stored in Buffer

    def has_atleast(self, size):
        return self.current_size >= size

    def sample(self, batch_size, nsteps):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]
            # self.current_size is the number of episodes stored in Buffer
            # So this is taking all the data in the buffer


        #buffers['o_2'] = buffers['o'][:, 1:, :] # [num_of_episodes, nsteps, 13] if 'o'
        #buffers['ag_2'] = buffers['ag'][:, 1:, :]

        drop_time_steps = None  # Have to think how to incorporate this
        transitions = self.sample_transitions(buffers, batch_size, nsteps) # this batch_size = nenvs. nsteps = number of steps in a trajectory
        # print("transitions: {}".format(transitions))

        print("transitions.keys() : {}".format(transitions.keys()))
        #new_transitions = list(set(['r', 'o_2', 'ag_2'] + list(self.buffers.keys())) - set('drop_time_steps'))
        print("self.buffers.keys():{}".format(self.buffers.keys()))
        #print("new transitions.keys() : {}".format(new_transitions))
        '''
        for key in (set(['r', 'o_2', 'ag_2'] + list(self.buffers.keys())) - set('drop_time_steps')):
            print("what is going on")
            print(key)
            assert key in transitions, "key %s missing from transitions" % key
        '''

        return transitions

    def sample_a2c(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_samples_a2c(idxes)

    def store_episode(self, episode_batch):
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        idxs = self._get_storage_idx(batch_size)
        for key in self.buffers.keys():
            if key == 'drop_time_steps':
                continue
            else:
                print('key: {}, batch shape: {}, buffer shape: {}'.format(key, episode_batch[key].shape, self.buffers[key][idxs].shape))
                print('idxs: {}, batch_size: {}'.format(idxs, batch_size))
                self.buffers[key][idxs] = episode_batch[key][0] #.reshape(batch_size, self.buffer_shapes[key])
        print("printing out episode_batch['drop']: {}".format(np.squeeze(episode_batch['drop'])))
        print("The index of 'drop' in episode {} is...".format(idxs))
        # TODO intergrate this drop thing with stuck
        if np.all(np.squeeze(episode_batch['drop'])==0) and np.all(np.squeeze(episode_batch['stuck'])==0): # checking if every element is zero
            print("Neither drop nor stuck didn't happen")
            self.buffers['drop_time_steps'][idxs] = self.num_rollouts # getting the index for HER drop_time_steps
        else:
            if not np.all(np.squeeze(episode_batch['drop'])==0):
                print("drop happened")
                self.buffers['drop_time_steps'][idxs] = np.where(np.squeeze(episode_batch['drop'])==1)[0][0]
            if not np.all(np.squeeze(episode_batch['stuck'])==0):
                print("stuck happened")
                self.buffers['drop_time_steps'][idxs] = np.where(np.squeeze(episode_batch['stuck'])==1)[0][0]
        # print("ggg: {}".format(np.where(np.squeeze(episode_batch['drop'])==1)))
        print("the index: {}".format(self.buffers['drop_time_steps'][idxs]))
        # self.n_transitions_stored += batch_size * self.T

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

    def _get_storage_idx(self, inc=None):
        inc = inc or 1 # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size: # in case inc is not 1
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
