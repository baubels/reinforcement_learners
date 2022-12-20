import time
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from utils import loss

from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss, update_policy, step_episode, initialise_episode

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

import gym
import matplotlib.pyplot as plt


from nets_utils import initialise_networks
from accessory import print_episode_info 

### custom libraries
import info
###


def train_DQN(env_name:str='CartPole-v1', exploration_schedule:list[int]=[10,1], network_layers:list[int]=[4,2], 
        episode_print_thresh:int=150, n_episodes:int=300, buffer_size=1000, batch_size=1, update_when=1, learning_rate=1, decay=0.99,
        recordings_dir_name:str='episode_recorder', episode_base_name:str='episode', record=False, DDQN_val=False):
    """Train a DQN Agent.

    Args:
        env_name (str, optional): `gym` environment to train agent on. Defaults to 'CartPole-v1'.
        exploration_schedule (list[int], optional): _description_. Defaults to [10,1].
        network_layers (list[int], optional): _description_. Defaults to [4,2].
        episode_print_thresh (int, optional): _description_. Defaults to 150.
        n_episodes (int, optional): _description_. Defaults to 300.
        buffer_size (int, optional): _description_. Defaults to 1000.
        batch_size (int, optional): _description_. Defaults to 1.
        update_when (int, optional): _description_. Defaults to 1.
        learning_rate (int, optional): _description_. Defaults to 1.
        decay (float, optional): _description_. Defaults to 0.99.
        recordings_dir_name (str, optional): _description_. Defaults to 'episode_recorder'.
        episode_base_name (str, optional): _description_. Defaults to 'episode'.
        record (bool, optional): _description_. Defaults to False.
        DDQN_val (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # ###
    # info.print_info('DQN', env_name, exploration_schedule, n_episodes, decay, network_layers, batch_size, buffer_size, update_when)
    # ###

    runs_results = []
    env = gym.make(env_name, render_mode='rgb_array')

    # loop through a run
    for run in range(exploration_schedule[0]):
        if record: recorder('new_run')                                             # <><><><1><><><> #

        # initialise networks and update scheme
        t0 = time.time()
        policy_net, target_net = initialise_networks(network_layers=[4,2])
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)          # using Adam gradient descent
        memory = ReplayBuffer(buffer_size)                                         # a replay buffer of size buffer_size

        # loop through episodes
        episode_durations = []
        for i_episode in range(n_episodes):
            print_episode_info(i_episode, n_episodes, episode_print_thresh)
            if record: video_recorder = recorder('start_episode')                  # <><><><2><><><> #

            # initialise episode starting state
            state, done, terminated, t = initialise_episode(env)

            # generate steps and update through an episode
            while not (done or terminated):
                if record: video_recorder.capture_frame()                          # <><><><3><><><> #

                # select action, observe results; push to memory
                action, next_state, reward, done, terminated = step_episode(env, policy_net, state, exploration_schedule[1], decay, i_episode)
                memory.push([state, action, next_state, reward, torch.tensor([done])])
                state = next_state

                # update the policy net
                update_policy(memory, policy_net, target_net, optimizer, 'DQN', batch_size)

                # check state termination
                if done or terminated: episode_durations.append(t+1)
                t += 1

            if record: recorder('end_episode', video_recorder)                     # <><><><4><><><> #

            # update the target net
            update_target(target_net, policy_net, i_episode, update_when)

        runs_results.append(episode_durations)
        t1 = time.time()
        print(f"Ending run {run+1} of {exploration_schedule[0]} with run time: {round(t1-t0, 2)} and average end episode length: {sum(episode_durations[-10:])/len(episode_durations[-10:])}")

    print('Complete')
    return runs_results, target_net