import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay"""
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size, nsteps):
        # Prepare episode_batch somwhere else so that it's more flexible to determine what to include in this
        # episode_batch (just to modularize the code) -> This is taken care of in def sample() in replay_buffer_her.py
        # episode_batch is {key: array(buffer_size x T x dim_key)}
        # I should take care of the drop case in the above prepare_episode_batch function.
        # drop_time_idxs should have shape [rollout_batch_size, 1]
        # drop_time_steps[i] is the time when the drop happens in the episode i (so it should be 0 <= t <= nsteps)
        # It looks like rollout_batch_size is the same thing as buffer_size

        # episode_batch is {key: array(buffer_size x T x dim_key)}
        # T is the number of rollouts, which should be determined by the batch_shape in ReplayBuffer class
        # This function should return a set of trajectories such that {key: array(nenv x nsteps x dim_key)}
        # where nenv is batch_size in our new setting.
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        # rollout_batch_size is the number of episodes that are stored in the buffer so far
        print("rollout_batch_size in her.py: {}".format(rollout_batch_size))
        #drop_time_steps = episode_batch['drop'][0][:batch_size]
        drop_time_steps = episode_batch['drop_time_steps']
        print("printing the shape of episode_batch['drop_time_steps']: {}".format(episode_batch['drop_time_steps']))
        drop_time_steps = drop_time_steps.reshape(-1)

        # Select which episodes and time steps (t_samples) to use
        ## t_samples
        ## randint(low, high, size) picks 'size' number of random integers from [low, high-1]
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        print("drop_time_steps: {}".format(drop_time_steps))
        print("ggg: {}".format(drop_time_steps[episode_idxs[0]]))
        print("ggg: {}".format(np.random.randint(drop_time_steps[episode_idxs[0]])))
        print("ggg: {}".format(np.random.randint(drop_time_steps[episode_idxs[i]]) for i in range(batch_size)))
        # If the drop id is smaller than n_steps, then I should just pick the first n_steps as a trajectory
        ### What does it mean for t_samples? Should I set t_samples to be the n_steps th id?
        ### I think the simplest answer is to replace the drop id with the n_steps if the drop id is smaller than the nsteps, and
        ### the corresponding t_sample should just be the nstep
        ### so that this opeartion future_offset = np.random.uniform(size=batch_size) * (drop_time_steps_subset - t_samples) makes sense. (there won't be no future_offset)
        nsteps += 1
        t_samples = []
        for i in range(batch_size):
            if drop_time_steps[episode_idxs[i]] <= nsteps:
                t_samples.append(nsteps)
                drop_time_steps[episode_idxs[i]] = nsteps
            else:
                t_samples.append(np.random.randint(low=nsteps, high=drop_time_steps[episode_idxs[i]]))
        t_samples = np.array(t_samples)
        print("drop_time_steps[episode_idxs[0]]".format(drop_time_steps[episode_idxs[0]]))
        drop_time_steps_subset = np.array([[drop_time_steps[episode_idxs[i]] for i in range(batch_size)]])
        drop_time_steps_subset = drop_time_steps_subset.reshape(-1)

        transitions = {}
        for key in episode_batch.keys():
            if key == 'drop_time_steps':
                continue
            dummy_ndarray = np.empty((episode_batch[key].shape[0],) + (nsteps,) + (episode_batch[key].shape[2],))
            print("dummy_ndarray.shape: {}".format(dummy_ndarray.shape))
            for i in range(batch_size):
                dummy_ndarray[i] = episode_batch[key][episode_idxs[i]][(t_samples[i]-nsteps):t_samples[i]].copy()
            transitions[key] = dummy_ndarray  #episode_batch[key][episode_idxs, (t_samples-nsteps):t_samples, :].copy()
        # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        print("t_samples.shape: {}".format(t_samples.shape))
        print("drop_time_steps_subset.shape: {}".format(drop_time_steps_subset.shape))
        print("drop_time_steps.shape: {}".format(drop_time_steps.shape))
        assert drop_time_steps_subset.shape == t_samples.shape
        future_offset = np.random.uniform(size=batch_size) * (drop_time_steps_subset - t_samples)
        future_offset = future_offset.astype(int)
        print("t_samples.shape: {}".format(t_samples.shape))
        print("future_offset.shape: {}".format(future_offset.shape))
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        print("episode_idx[her_indexes]: {}, future_t: {}".format(episode_idxs[her_indexes], future_t))
        print("episode_idx[her_indexes].shape: {}, future_t.shape: {}".format(episode_idxs[her_indexes].shape, future_t.shape))
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        # The goal should be the same for the entire trajectory
        print("transitions['g'][her_indexes][i, j].shape: {}".format(transitions['g'][her_indexes][0, 0].shape))
        print("future_ag[i].shape: {}".format(future_ag[0].shape))
        #for i in range(batch_size): # The number of episodes I might update is not necessarily the entire batch. so this is wrong
        print("her_indexes: {}".format(her_indexes))
        print("her_indexes length: {}".format(her_indexes[0].shape[0]))
        for i in range(her_indexes[0].shape[0]):
            for j in range(nsteps):
                transitions['g'][her_indexes][i, j] = future_ag[i]
        #transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        # for key, value in transitions.items():
        #    if key.startswith('info_'):
        #        info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        # reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params = {k: transitions[k] for k in ['ag', 'g']}
        # reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)
        transitions['r'] = np.expand_dims(transitions['r'], 2)

        for k in transitions.keys():
            print(k)
            print(transitions[k].shape)

        #transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
        #               for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size)

        return transitions

    return _sample_her_transitions