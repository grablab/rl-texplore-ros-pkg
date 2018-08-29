import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay"""
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size):
        # Prepare episode_batch somwhere else so that it's more flexible to determine what to include in this
        # episode_batch (just to modularize the code) -> This is taken care of in def sample() in replay_buffer_her.py
        # episode_batch is {key: array(buffer_size x T x dim_key)}
        # I should take care of the drop case in the above prepare_episode_batch function.
        # drop_time_idxs should have shape [rollout_batch_size, 1]
        # drop_time_steps[i] is the time when the drop happens in the episode i (so it should be 0 <= t <= nsteps)
        # It looks like rollout_batch_size is the same thing as buffer_size
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        print("rollout_batch_size in her.py: {}".format(rollout_batch_size))
        drop_time_steps = episode_batch['drop'][0][:batch_size]
        drop_time_steps = np.squeeze(drop_time_steps)

        # Select which episodes and time steps (t_samples) to use
        ## t_samples
        ## randint(low, high, size) picks 'size' number of random integers from [low, high-1]
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        print("drop_time_steps: {}".format(drop_time_steps))
        print("ggg: {}".format(drop_time_steps[episode_idxs[0]]))
        print("ggg: {}".format(np.random.randint(drop_time_steps[episode_idxs[0]])))
        print("ggg: {}".format(np.random.randint(drop_time_steps[episode_idxs[i]]) for i in range(batch_size)))
        t_samples = np.array([np.random.randint(drop_time_steps[episode_idxs[i]]) for i in range(batch_size)])
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        print("t_samples.shape: {}".format(t_samples.shape))
        print("drop_time_steps.shape: {}".format(drop_time_steps.shape))
        assert drop_time_steps.shape == t_samples.shape
        future_offset = np.random.uniform(size=batch_size) * (drop_time_steps - t_samples)
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
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        # for key, value in transitions.items():
        #    if key.startswith('info_'):
        #        info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        # reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        for k in transitions.keys():
            print(k)
            print(transitions[k].shape)

        #transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
        #               for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size)

        return transitions

    return _sample_her_transitions