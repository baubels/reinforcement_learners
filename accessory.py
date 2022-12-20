import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import torch
import numpy as np


def print_episode_info(episode:int, n_episodes:int, episode_threshold:int=100):
    if (episode+1) % episode_threshold == 0:
        print("episode ", episode+1, "/", n_episodes)

def print_training_info(name:str, env_name, n_runs, starting_eps, n_episodes, decay:float, network_layers:list[int], batch_size, buffer_size, update_when)->None:
    print(f"TRAINING!!! A {name} agent on the {env_name} environment over {n_runs} runs each with {n_episodes} episodes.")
    print(f"Episodes are generated with an eps-greedy policy with eps = {starting_eps}, decaying at eps*{decay}^episode_count")
    print(f"Each policy and target DQN net has feedforward layer widths: {network_layers}.\n")
    print(f"Backpropogation is done with SGD with batchsize {batch_size} sampled over a buffer of size {buffer_size} and updating policy network every {update_when} episodes.")


def recorder(action:str, video_recorder=None, video_dir=None, recordings_dir_name:str=None, run:int=None, episode_base_name:str=None, env=None, i_episode:int=None)->None:
    """Records the Agent learning according to `action`.

    Args:
        action: One of 'new_run', 'start_episode', 'end_episode'. Denotes a recording done according to agent progress.
    """
    if action == 'new_run':
        cwd = os.getcwd()
        video_dir = os.path.join(cwd, recordings_dir_name + f'run_{run}')
        if not os.path.isdir(video_dir): os.mkdir(video_dir)
        return video_dir

    elif action == 'start_episode':
        # <><><><><><><> #
        video_file = os.path.join(video_dir, episode_base_name + f"{i_episode}.mp4")
        video_recorder = VideoRecorder(env, video_file, enabled=True)  #record a video of the episode
        # <><><><><><><> #
        return video_recorder
    
    elif action == 'end_episode':
        video_recorder.capture_frame()
        video_recorder.close()
        video_recorder.enabled = False

def print_results(runs_results, n_episodes:int=300, ylabel='return', xlabel='episode', title='title'):
    """Prints the episode value results of a trained DQN season."""
    plt.figure(figsize=(20,5))
    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    stds = results.float().std(0)
    plt.plot(torch.arange(n_episodes), means)
    plt.fill_between(np.arange(n_episodes), means, means+stds, alpha=0.3, color='b')
    plt.fill_between(np.arange(n_episodes), means, means-stds, alpha=0.3, color='b')
    plt.axhline(y=100, color='r', linestyle='--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def print_results_several(runs_results_list, n_episodes:int=300, ylabel='return', xlabel='episode', title='title'):
    """Prints the episode value results of a 2 run sets. Includes colorbars."""
    plt.figure(figsize=(20,5))
    i = 0
    netnames = ['DQN', 'DDQN']
    colors=['blue','red']
    for runs_results in runs_results_list:
        plt.title(title)
        results = torch.tensor(runs_results)
        means = results.float().mean(0)
        stds = results.float().std(0)
    
        plt.plot(torch.arange(n_episodes), means, label=netnames[i])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.fill_between(np.arange(n_episodes), means, means+stds, alpha=0.3,color=colors[i], label=netnames[i])
        plt.fill_between(np.arange(n_episodes), means, means-stds, alpha=0.3,color=colors[i])
        i += 1
    plt.legend()
    plt.axhline(y=100, color='r', linestyle='--')
    plt.show()


def print_batch(batch_runs, ylabel:str, xlabel:str, title:str, legends:list[str]):
    """Prints the results of hyperparameter tuning across different hyperparameters."""
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    for br in range(len(batch_runs)):
        results = torch.tensor(batch_runs[br])
        means = results.float().mean(0)
        stds = results.float().std(0)
        plt.plot(torch.arange(300), means, label=str(legends[br]))
        plt.fill_between(np.arange(300), means, means+stds, alpha=0.1)
        plt.fill_between(np.arange(300), means, means-stds, alpha=0.1)
        plt.legend()
    plt.show()
