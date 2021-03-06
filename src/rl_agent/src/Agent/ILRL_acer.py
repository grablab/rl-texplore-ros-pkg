#!/usr/bin/python
import os
import sys
import numpy as np
import rospy
from common.policies import build_policy
from common.rollout import RolloutWorker
from acer import Model, Acer
from collections import deque

NODE = "RLAgent"
MODE = 'acer_mlp' # 'il'  # nfq
# MODEL_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/projects/sliding_policies/models/' + MODE + '/')
MODEL_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/')


def train(model, rollout_worker, n_epochs, n_batches, demo_file):
    Q_history = deque()
    q_hist, critic_loss_hist, actor_loss_hist, ent_hist, bc_loss_hist = [], [], [], [], []
    if model.bc_loss == 1:
        model.initDemoBuffer(demo_file)
    for epoch in range(n_epochs):
        #print('ok')
        if rollout_worker.compute_Q:
            episode, mean_Q = rollout_worker.generate_rollouts()
        else:
            episode = rollout_worker.generate_rollouts()
        # TODO Check how store_episode will go
        model.store_episode(episode)
        critic_loss_que, actor_loss_que, ent_que = [], [], []
        bc_loss_que = []
        for i in range(n_batches): # update q-values
            critic_loss, actor_loss, ent, bc_loss_np = model.train()
            critic_loss_que.append(critic_loss); actor_loss_que.append(actor_loss)
            ent_que.append(ent); bc_loss_que.append(bc_loss_np)
            # print("n_batch: {}, critic_loss: {}, actor_loss: {}".format(i, critic_loss, actor_loss))
        print("Mean Q-value: {}".format(mean_Q))
        mean_critic_loss = np.mean(critic_loss_que)
        mean_actor_loss = np.mean(actor_loss_que)
        mean_ent = np.mean(ent_que)
        mean_bc_loss = np.mean(bc_loss_que)
        print("Mean critic loss: {}".format(mean_critic_loss))
        print("Mean actor loss: {}".format(mean_actor_loss))
        print("Mean bc loss: {}".format(mean_bc_loss))
        print("Mean entopy: {}".format(mean_ent))
        q_hist.append(mean_Q)
        critic_loss_hist.append(mean_critic_loss)
        actor_loss_hist.append(mean_actor_loss)
        ent_hist.append(mean_ent)
        bc_loss_hist.append(mean_bc_loss)
        #model.update_target_net() # update the target net less frequently
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/q_val.npy', np.array(q_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/cri_loss.npy', np.array(critic_loss_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/actor_loss.npy', np.array(actor_loss_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/ent.npy', np.array(ent_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/bc_loss.npy', np.array(bc_loss_hist))
        save_loc = model.save_model()
        print('saved model at : {} after {} epochs'.format(save_loc, epoch+1))

def learn(model, runner, nenvs, nsteps, replay_start, replay_ratio, total_timesteps):
    log_interval = 10 # this is not used for now
    acer = Acer(runner, model, log_interval)
    nbatch = nenvs*nsteps
    for acer.steps in range(0, total_timesteps, nbatch):
        acer.call(on_policy=True)  # TODO: Modify generate_rollout in rollout.py for HER.
        if model.buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no rollout with T42 in this
    return model

def save_episode(episode, i):
    obs = episode['o']
    print(obs)
    SAVE_DIR = "/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/data-init-goal/original"
    file_path = os.path.join(SAVE_DIR, 'episode_'+str(i)+'.out')
    obs_reshaped = np.squeeze(obs)
    np.savetxt(file_path, obs_reshaped, delimiter=',')

def save_episode_keyboard(episode, i):
    obs = episode['o'][0]
    actions = episode['u'][0]
    print("In save_episode_keyboard...obs.shape: {}, u.shape: {}".format(obs.shape, actions.shape))
    obs_actions_combined = np.concatenate((obs, actions), axis=1)
    SAVE_DIR = "/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/data-init-goal/keyboard_demo"
    file_path = os.path.join(SAVE_DIR, 'episode_'+str(i)+'.out')
    obs_actions_combined = np.squeeze(obs_actions_combined)
    np.savetxt(file_path, obs_actions_combined, delimiter=',')


def collect_random_rl_data(runner):
    i = 1
    while True:
        print("Episode: {}".format(i))
        episode = runner.generate_rollouts()
        if episode['drop'][0][0][0] == 1 or episode['stuck'][0][0][0] == 1:
            print("Drop happened at time step 1. Ignoring this episode...")
            continue
        save_episode(episode, i)
        i += 1

def record_demonstration_w_keyboard(runner):
    i = 1
    while True:
        print("Episode: {}".format(i))
        # Do I need to limit the number of rollouts?
        episode = runner.generate_rollouts_w_keyboard()
        if episode['drop'][0][0][0] == 1 or episode['stuck'][0][0][0] == 1:
            print("Drop happened at time step 1. Ignoring this episode...")
            continue
        save_episode_keyboard(episode, i)
        i += 1


if __name__ == '__main__':
    rospy.init_node(NODE)
    rospy.loginfo('started RLAgent node')
    dims = {'o': 13, 'u': 9}
    model_name = 'il_policy_for_a2c' #'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1'
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, model_name)
    n_epochs = 100000
    random_eps = 0.1
    bc_loss = True #False
    nenvs = 2 # the batch size in training. Something we can treat in parallel.
    nsteps = 40 # This should just be the number of steps in a sampled trajectories. Should be dealt in def call() in Acer()
    batch_size = 40      # Check if these guys matter at all in this new setup; might be from old scripts
    # This acutually matters for ReplayBuffer. This should be the number of steps in a sampled trajectory
    demo_batch_size = 40 # Check if these guys matter at all in this new setup; might be from old scripts
    # bc_loss = True # See def configure_mlp in config.py too
    num_rollouts = 50 # This is a new variable. Should only be used in rollout.py and ReplayBuffer. Remember that the T as in the episode batch shape is this num_rollouts.
    network = 'mlp'
    network_kargs = {}
    policy = build_policy(network, estimate_q=True, **network_kargs)
    #TODO: Test this ILRL_acer.py upto here now that I changed the observation dimension.

    '''
    def __init__(self, policy, num_states, num_actions, nenvs, nsteps,
                 ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta): 
    '''

    ent_coef =0.01 ; q_coef=0.5 ; gamma=0.99 ; max_grad_norm=10 ;
    rprop_alpha=0.99 ; rprop_epsilon=1e-5 ; total_timesteps=int(10e5) ;
    trust_region=False; alpha=0.99 ; delta =1 ; # what are alpha and delta again?
    lrschedule='linear' ; c =10.0 ; lr =7e-4
    buffer_size=10000; bc_loss=False
    model = Model(policy, num_states=dims['o'], num_actions=dims['u'], nenvs=nenvs, nsteps=nsteps, num_rollouts=num_rollouts,
                  ent_coef=ent_coef, q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr,
                  rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule, c=c, trust_region=trust_region, alpha=alpha, delta=delta,
                  buffer_size=buffer_size, bc_loss=bc_loss)
#                  bc_loss=bc_loss,
#                  batch_size=40, demo_batch_size=20,
#                  model_name=model_name, save_path=MODEL_SAVE_PATH, checkpoint_path=checkpoint_path, restore=True)
    print('gggggggggggg')
    print(model)
    print('gggggggggggg')

    record_demo_data_w_keyboard=True
    rollout_worker = RolloutWorker(model, dims, num_rollouts, use_target_net=True, compute_Q=True, random_eps=random_eps, record_demo_data_w_keyboard=True)

    # n_batches = 10 #2
    demo_file = '/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/data/demodata.npy'
    #train(model=model, rollout_worker=rollout_worker, n_epochs=n_epochs, n_batches=n_batches, demo_file=demo_file)
    replay_start = 2 # 1000  # int, the sampling from the replay buffer does not start until replay buffer has at least that many samples
    replay_ratio = 4  # int, how many (on averages) batches of data fo sample from the replay buffer take after batch from the environment

    #collect_random_rl_data(runner=rollout_worker)
    record_demonstration_w_keyboard(runner=rollout_worker)

    '''
    learn(model=model, runner=rollout_worker, nenvs=nenvs, nsteps=nsteps, replay_start=replay_start,
         replay_ratio=replay_ratio, total_timesteps=total_timesteps)
    '''