"""
File implementing:
    `initialise_episode`: initialise an episode (pygame env instance) with a random state
    `step_episode`: Do a step according to an epsilon-greedy policy within a pygame env instance using a DQN network.
    ...
"""

import random
from collections import deque
import torch
import torch.nn.functional as F

# from nets_utils import DQN


def initialise_episode(env) -> tuple[torch.Tensor, bool, bool, int]:
    """Initialise an episode with random state.

    Args:
        env: A playable game environment instance able to generate states, if done, terminated, and step of episode info.
             (Note: the snake environment has been coded up to work seemlessly with this.)

    Returns:
        state: An initial state.
        done: An indication on whether the environment is done (episode completion, time limit exceeded, physics, etc.)
        terminated: A boolean indicating termination.
        t: An int of episode count. Episodes are counted from 0.
    """
    observation, _ = env.reset()
    state = torch.tensor(observation).float()
    done, terminated, t = False, False, 0
    return state, done, terminated, t


def step_episode(env, policy_net, state, eps: float, decay: float, episode: int, kind='DQN'):
    """Do a single step in an episode.

    Args:
        env (): An OpenAI gym environment instance
        policy_net: A deep neural network of type `kind`
        state: _description_
        eps: value of epsilon in an eps-greedy policy
        decay: A decay parameter decaying epsilon according to the episode count
        episode: The episode count
        kind: One of 'A2C', 'REINFORCE', 'DQN', 'DDQN'.
              A change indicates change in how actions are made and output types.

    Returns:
        action: An made by the agent.
        next_state: A next state received by the agent (includes environmental noise P).
        reward: A reward signal received for arriving at next_state.
        done: A boolean indicating whether the environment terminated (physics, time limit, terminal state reached).
        terminated: A boolean indicating whether the terminal state has been reached.
        (if kind is one of 'A2C' or 'REINFORCE')
        log_prob_action ('A2C' or 'REINFORCE'): A log probability of the action chosen. (log(action))
        value ('A2C'):  A baseline state-value of the initial state.
    """

    if kind == 'A2C':  # policy-based + state-value method (acting as a baseline)
        log_prob_action, action, value = policy_net(state)
        observation, reward, done, terminated, _ = env.step(action)
        # print('observation: {}, reward: {}, done: {}'.format(observation, reward, done))
        reward, action = torch.tensor([reward]), torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()
        return action, next_state, reward, done, terminated, log_prob_action, value

    elif kind == 'REINFORCE':  # a policy-based method
        log_prob_action, action = policy_net(state)
        observation, reward, done, terminated, _ = env.step(action)
        reward, action = torch.tensor([reward]), torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()
        return action, next_state, reward, done, terminated, log_prob_action

    elif kind in ['DQN', 'DDQN']:  # a value-action based method
        from nets_utils import epsilon_greedy

        # determine an epsilon-greedy action
        action = epsilon_greedy(eps * (decay ** episode), policy_net, state)

        # do a step in the environment; convert values to Torch.tensor     
        observation, reward, done, terminated, _ = env.step(action)
        reward, action = torch.tensor([reward]), torch.tensor([action])
        next_state = torch.tensor(observation).reshape(-1).float()
        return action, next_state, reward, done, terminated


class ReplayBuffer:
    def __init__(self, size: int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)

    def push(self, transition) -> deque:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size: int) -> list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

################################################################################
################################################################################
