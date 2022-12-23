from general_utils import *
from nets_utils import *
from accessory import print_episode_info, recorder, print_training_info

import time
import torch
import torch.optim as optim
from gyms.cartpole import CartPoleEnv
from gyms.acrobot import AcrobotEnv
from gyms.mountain_car import MountainCarEnv



def train_DQN(type:str='DQN', env_name:str='CartPole', n_runs:int=10, starting_eps:float=1., network_layers:list[int]=[4,2], 
        episode_print_thresh:int=150, n_episodes:int=300, buffer_size=1000, batch_size=1, update_when=1, learning_rate=1, decay=0.99,
        recordings_dir_name:str='episode_recorder', episode_base_name:str='episode', record=False, max_episode_steps=500, lengths_to_consider=1):
    """Train a DQN or DDQN pair of networks according to some pygame env instance.

    Args:
        type (str, optional): One of `DQN` or `DDQN`. Defaults to 'DQN'.
        env_name (str, optional): One of `CartPoleEnv`, `AcrobotEnv`, `MountainCarEnv`. Defaults to 'CartPole'. See `gyms/environment_info.txt` for usage info.
        n_runs (int, optional): Number of runs to do the algorithm on. Defaults to 10.
        starting_eps (float, optional): Starting epsilon exploration parameter for an eps-greedy policy. Defaults to 1..
        network_layers (list[int], optional): The feed-forward neural network doing the learning. Defaults to [4,2].
        episode_print_thresh (int, optional): A number denoting the number of episodes to run before printing episode count info. Defaults to 150.
        n_episodes (int, optional): The number of episodes to use in a run. Defaults to 300.
        buffer_size (int, optional): The memory size ((s,a,r,s)) to have of past game states. Defaults to 1000.
        batch_size (int, optional): The batch size in SGD for learning the network. Defaults to 1.
        update_when (int, optional): A number indicating after how many episodes to update the target network from the policy one. Defaults to 1.
        learning_rate (int, optional): The initial learning rate to use in learning the networks. Defaults to 1.
        decay (float, optional): The epsilon decay parameter to use. Throughout learning, eps = eps_0*decay**episode. Defaults to 0.99.
        recordings_dir_name (str, optional): The directory to store recorded episodes - if set to be True. Defaults to 'episode_recorder'.
        episode_base_name (str, optional): The starting file name of recorded episodes. Defaults to 'episode'.
        record (bool, optional): A true or false value denoting whether episodes should be recorded. Defaults to False.
        max_episode_steps (None, optional): The maximal number of steps to have per episode. If None, the openai default for the env is used. Defaults to None.

    Returns:
        runs_results: A 2d list of a list of durations per episode per run
        target_net: A trained DQN network according to the last run
    """
    print_training_info(type, env_name, n_runs, starting_eps, n_episodes, decay, network_layers, batch_size, buffer_size, update_when)
    runs_results = []

    env = eval(env_name)(render_mode='rgb_array')
    
    # loop through a run
    for run in range(n_runs):
        if record: video_dir = recorder('new_run', recordings_dir_name=recordings_dir_name, run=run)   # <><><><1><><><> #

        # initialise networks and update scheme
        t0 = time.time()
        policy_net, target_net = initialise_networks(network_layers)
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)          # using Adam gradient descent
        memory = ReplayBuffer(buffer_size)                                         # a replay buffer of size buffer_size
        # early_stopping=0
        # loop through episodes
        episode_durations = []
        for i_episode in range(n_episodes):
            print_episode_info(i_episode, n_episodes, episode_print_thresh, sum(episode_durations[-lengths_to_consider:])/lengths_to_consider)
            # print_episode_info(i_episode, n_episodes, episode_print_thresh)
            if record: video_recorder = recorder('start_episode', episode_base_name=episode_base_name, video_dir=video_dir, env=env, i_episode=i_episode)   # <><><><2><><><> #

            # initialise episode starting state
            state, done, terminated, t = initialise_episode(env)
            counter = 0
            # generate steps and update through an episode
            while not (done or terminated) and counter < max_episode_steps:
                counter += 1
                if record: video_recorder.capture_frame()                                  # <><><><3><><><> #

                # select action, observe results; push to memory
                action, next_state, reward, done, terminated = step_episode(env, policy_net, state, starting_eps, decay, i_episode, kind=type)
                memory.push([state, action, next_state, reward, torch.tensor([done])])
                state = next_state

                # update the policy net
                update_policy(memory, policy_net, target_net, optimizer, type, batch_size)

                # check state termination
                t += 1
            episode_durations.append(t+1)
            if record: recorder('end_episode', video_recorder)                     # <><><><4><><><> #

            # update the target net
            update_target(target_net, policy_net, i_episode, update_when)

            # # early stopping if the policy is too good; this ruins some of the print functionalities
            # if episode_durations[-1] == max_episode_steps + 1:
            #     early_stopping += 1
            # if early_stopping >= 10:
            #     print('stopping early as the net is consistently {} achieving the maximal episode length goal of {}'.format(10, max_episode_steps+1))
            #     break

        runs_results.append(episode_durations)
        t1 = time.time()
        print(f"Ending run {run+1} of {n_runs} with run time: {round(t1-t0, 2)} and average end episode length: {sum(episode_durations[-10:])/len(episode_durations[-10:])}")

    print('Complete')
    return runs_results, target_net


def train_REINFORCE(env_name:str='CartPoleEnv', n_runs:int=10, starting_eps:float=1., network_layers:list[int]=[4,2], 
        episode_print_thresh:int=150, n_episodes:int=300, batch_size=1, update_when=1, learning_rate=1, decay=0.99,
        recordings_dir_name:str='episode_recorder', episode_base_name:str='episode', record=False, max_episode_steps=500, discount:float=0.7, lengths_to_consider=1):
    """Train on some pygame env instance via REINFORCE.

    Args:
        env_name (str, optional): One of `CartPoleEnv`, `AcrobotEnv`, `MountainCarEnv`. Defaults to 'CartPole'. See `gyms/environment_info.txt` for usage info.
        n_runs (int, optional): Number of runs to do the algorithm on. Defaults to 10.
        starting_eps (float, optional): Starting epsilon exploration parameter for an eps-greedy policy. Defaults to 1..
        network_layers (list[int], optional): The feed-forward neural network doing the learning. Defaults to [4,2].
        episode_print_thresh (int, optional): A number denoting the number of episodes to run before printing episode count info. Defaults to 150.
        n_episodes (int, optional): The number of episodes to use in a run. Defaults to 300.
        batch_size (int, optional): The batch size in SGD for learning the network. Defaults to 1.
        update_when (int, optional): A number indicating after how many episodes to update the target network from the policy one. Defaults to 1.
        learning_rate (int, optional): The initial learning rate to use in learning the networks. Defaults to 1.
        decay (float, optional): The epsilon decay parameter to use. Throughout learning, eps = eps_0*decay**episode. Defaults to 0.99.
        recordings_dir_name (str, optional): The directory to store recorded episodes - if set to be True. Defaults to 'episode_recorder'.
        episode_base_name (str, optional): The starting file name of recorded episodes. Defaults to 'episode'.
        record (bool, optional): A true or false value denoting whether episodes should be recorded. Defaults to False.
        max_episode_steps (None, optional): The maximal number of steps to have per episode. If None, the openai default for the env is used. Defaults to None.

    Returns:
        runs_results: A 2d list of a list of durations per episode per run
        policy_net: A trained REINFORCE network according to the last run
    """
    print_training_info('REINFORCE', env_name, n_runs, starting_eps, n_episodes, decay, network_layers, batch_size, max_episode_steps, update_when)
    runs_results = []

    env = eval(env_name)(render_mode='rgb_array')
    


    # loop through a run
    for run in range(n_runs):
        if record: video_dir = recorder('new_run', recordings_dir_name=recordings_dir_name, run=run)   # <><><><1><><><> #

        # initialise networks and update scheme
        t0 = time.time()
        policy_net = REINFORCE(network_layers)
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)          # using Adam gradient descent
        # memory = ReplayBuffer(max_episode_steps)                                         # a replay buffer of size buffer_size

        # loop through episodes
        episode_durations = []
        for i_episode in range(n_episodes):
            print_episode_info(i_episode, n_episodes, episode_print_thresh, sum(episode_durations[-lengths_to_consider:])/lengths_to_consider)
            if record: video_recorder = recorder('start_episode', episode_base_name=episode_base_name, video_dir=video_dir, env=env, i_episode=i_episode)   # <><><><2><><><> #
            
            states, actions, rewards, log_probs = [], [], [], []
            
            # episode start
            state, done, terminated, t = initialise_episode(env)
            counter = 0
            # generate steps and update through an episode
            while not (done or terminated) and counter < max_episode_steps:
                counter += 1
                if record: video_recorder.capture_frame()                                  # <><><><3><><><> #
                
                # select action, observe results; push to memory
                action, next_state, reward, done, terminated, log_prob = step_episode(env, policy_net, state, starting_eps, decay, i_episode, kind='REINFORCE')
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state

                # check state termination
                t += 1
            
            episode_durations.append(t+1)
            # an episode is now in memory... update the net!
            discounted_returns = compute_discounted_returns(states, rewards, discount)

            # backprop - knowing a good learning rate here would be great!
            log_prob = torch.stack(log_probs)
            policy_gradient = -log_prob*discounted_returns  # theta_{t+1} = theta{t} + lr*discounted_returns*grad[ln(pi(A_t|S_t, theta_t))]

            policy_net.zero_grad()
            policy_gradient.sum().backward()
            optimizer.step()
                
            if record: recorder('end_episode', video_recorder)                     # <><><><4><><><> #


        runs_results.append(episode_durations)
        t1 = time.time()
        print(f"Ending run {run+1} of {n_runs} with run time: {round(t1-t0, 2)} and average end episode length: {sum(episode_durations[-10:])/len(episode_durations[-10:])}")

    print('Complete')
    return runs_results, policy_net

def train_ACTOR_CRITIC(env_name:str='CartPoleEnv', n_runs:int=10, starting_eps:float=1., network_layers:list[int]=[4,2], 
        episode_print_thresh:int=150, n_episodes:int=300, batch_size=1, update_when=1, learning_rate=1, decay=0.99,
        recordings_dir_name:str='episode_recorder', episode_base_name:str='episode', record=False, max_episode_steps=500, discount:float=0.7, lengths_to_consider:int=10):
    """Train on some pygame env instance via ACTOR-CRITIC. The update scheme specifically uses the advantage function as a baselined discounted return.
    ie)  theta_{t+1} = theta_{t} + lr*advantage*grad[ln(pi(A_t|S_t, theta_t))] (works like a gradient-method)

    Args:
        env_name (str, optional): One of `CartPoleEnv`, `AcrobotEnv`, `MountainCarEnv`. Defaults to 'CartPole'. See `gyms/environment_info.txt` for usage info.
        n_runs (int, optional): Number of runs to do the algorithm on. Defaults to 10.
        starting_eps (float, optional): Starting epsilon exploration parameter for an eps-greedy policy. Defaults to 1..
        network_layers (list[int], optional): The feed-forward neural network doing the learning. Defaults to [4,2].
        episode_print_thresh (int, optional): A number denoting the number of episodes to run before printing episode count info. Defaults to 150.
        n_episodes (int, optional): The number of episodes to use in a run. Defaults to 300.
        batch_size (int, optional): The batch size in SGD for learning the network. Defaults to 1.
        update_when (int, optional): A number indicating after how many episodes to update the target network from the policy one. Defaults to 1.
        learning_rate (int, optional): The initial learning rate to use in learning the networks. Defaults to 1.
        decay (float, optional): The epsilon decay parameter to use. Throughout learning, eps = eps_0*decay**episode. Defaults to 0.99.
        recordings_dir_name (str, optional): The directory to store recorded episodes - if set to be True. Defaults to 'episode_recorder'.
        episode_base_name (str, optional): The starting file name of recorded episodes. Defaults to 'episode'.
        record (bool, optional): A true or false value denoting whether episodes should be recorded. Defaults to False.
        max_episode_steps (None, optional): The maximal number of steps to have per episode. If None, the openai default for the env is used. Defaults to None.

    Returns:
        runs_results: A 2d list of a list of durations per episode per run
        policy_net: A trained REINFORCE network according to the last run
    """
    print_training_info('A2C', env_name, n_runs, starting_eps, n_episodes, decay, network_layers, batch_size, max_episode_steps, update_when)
    runs_results = []

    env = eval(env_name)(render_mode='rgb_array')
    
    # loop through a run
    for run in range(n_runs):
        if record: video_dir = recorder('new_run', recordings_dir_name=recordings_dir_name, run=run)   # <><><><1><><><> #

        # initialise networks and update scheme
        t0 = time.time()
        policy_net = ACTOR_CRITIC(network_layers) # in A2C, to change parameters just change the code
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)          # using Adam gradient descent
        # memory = ReplayBuffer(max_episode_steps)                                         # a replay buffer of size buffer_size

        # loop through episodes
        episode_durations = []
        for i_episode in range(n_episodes):
            print_episode_info(i_episode, n_episodes, episode_print_thresh, sum(episode_durations[-lengths_to_consider:])/lengths_to_consider)
            if record: video_recorder = recorder('start_episode', episode_base_name=episode_base_name, video_dir=video_dir, env=env, i_episode=i_episode)   # <><><><2><><><> #
            
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            # episode start
            state, done, terminated, t = initialise_episode(env)
            
            counter = 0
            # generate steps and update through an episode
            while not (done or terminated) and counter < max_episode_steps:
                counter += 1
                if record: video_recorder.capture_frame()                                  # <><><><3><><><> #

                # select action, observe results; push to memory
                action, next_state, reward, done, terminated, log_prob, value = step_episode(env, policy_net, state, starting_eps, decay, i_episode, kind='A2C')
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                state = next_state

                # check state termination
                t += 1
            
            episode_durations.append(t+1)
            # an episode is now in memory... update the net!
            discounted_returns = compute_discounted_returns(states, rewards, discount)
            values = torch.stack(values)

            # backprop - knowing a good learning rate here would be great!
            advantage = discounted_returns - values # Q estimate - V estimate acting as the baseline
            log_prob = torch.stack(log_probs)
            actor_loss = -log_prob*advantage    # theta_{t+1} = theta{t} + lr*advantage*grad[ln(pi(A_t|S_t, theta_t))] (works like a gradient-method)
            critic_loss = (discounted_returns-values)**2
            policy_gradient = actor_loss + critic_loss

            policy_net.zero_grad()
            policy_gradient.sum().backward()
            optimizer.step()
            
            if record: recorder('end_episode', video_recorder)                     # <><><><4><><><> #


        runs_results.append(episode_durations)
        t1 = time.time()
        print(f"Ending run {run+1} of {n_runs} with run time: {round(t1-t0, 2)} and average end episode length: {sum(episode_durations[-10:])/len(episode_durations[-10:])}")

    print('Complete')
    return runs_results, policy_net
