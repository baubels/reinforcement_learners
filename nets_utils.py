import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def safe_log(x):
    eps=1e-7
    x = F.relu(x)
    return torch.log(x+eps)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int], activation='F.relu'):
        """
        DQN initialisation. This is just a feedforward net with relu activations.

        Args:
            layer_sizes: list with size of each layer as elements
        """
        self.activation = activation
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = eval(self.activation)(layer(x))
        return x

    
class REINFORCE(nn.Module):
    def __init__(self, layer_sizes:list[int], activation='F.relu'):
        """
        REINFORCE initialisation. This is just a feedforward net with relu activations.

        Args:
            layer_sizes: list with size of each layer as elements
        """
        self.activation = activation
        self.layer_sizes = layer_sizes
        super(REINFORCE, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """***
        
        Returns:
            log_prob_action: the log probabilities of the outputs
            action: the action probabilities (output) of the REINFORCE policy net.
        """
        for layer in self.layers:
            x = eval(self.activation)(layer(x))
        
        actions = F.softmax(x, dim=0)
        action = self.get_action(actions)
        log_prob_action = safe_log(actions.squeeze(0))[action]
        
        return log_prob_action, action
    
    def get_action(self,a):
        return np.random.choice(np.arange(self.layer_sizes[-1]), p=a.squeeze(0).detach().cpu().numpy())

class ACTOR_CRITIC(nn.Module):
    def __init__(self, layer_sizes:list[int]=[4,32,2], activation='F.relu') -> None:
        # doing actor-critic with just a single hidden layer; actors and critics sharing this layer
        super().__init__()
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.actor = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.critic = nn.Linear(layer_sizes[1], 1)

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = eval(self.activation)(self.layer_1(x))
        actions = F.softmax(self.actor(x), dim=0)
        action = self.get_action(actions)
        log_prob_action = safe_log(actions.squeeze(0))[action]
        # return action values and critic
        return log_prob_action, action, self.critic(x)

    def get_action(self,a):
        return np.random.choice(np.arange(self.layer_sizes[-1]), p=a.squeeze(0).detach().cpu().numpy())

    
def initialise_networks(network_layers:list[int])->None:
    """Initialise policy and target networks.

    Args:
        network_layers (list[int]): _description_

    Returns:
        _type_: _description_
    """
    
    policy_net = DQN(network_layers)                                           # policy net of layers network_layers
    target_net = DQN(network_layers)                                           # target net of layers network_layers
    update_target(target_net, policy_net)                                      # load target_net with parameters of policy_net
    target_net.eval()                                                          # set the target net to a no-learning (eval) mode
    return policy_net, target_net

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon: return greedy_act
    else: return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN, episode:int=None, threshold:int=None):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """
    if episode and threshold:
        if episode % threshold == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())
    
    else:
        target_dqn.load_state_dict(policy_dqn.state_dict())

def update_policy(memory, policy_net:DQN, target_net:DQN, optimizer:torch.optim, network_type:str, batch_size:int):
    """Update policy network parameters using a gradient descent (or related) scheme.
    Does not return anything but modifies the policy network passed as parameter

    Args:
        memory (ReplayBuffer): A Replay Buffer consisting of past episode steps as memory.
        policy_net: A DQN that chiefly enabled the trajectory policy.
        target_net: A DQN that is used as a baseline for updating the policy network.
        optimizer: The optimizer used to apply a gradient-descent-like scheme.
        network_type: A string, one of 'DQN' or 'DDQN'. Directly influences the loss-function used.
        batch_size: Batch size to use for SGD. Self-explanatory.
    
    Returns: None
        A parameter-updated policy network according to one step of gradient-descent (or same scheme)
    """
    if not len(memory.buffer) < batch_size:               # update only if the memory buffer is at least has batch_size number of state action values in it
        transitions = memory.sample(batch_size)           # sample
        state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions)) # unpack and stack memory
        
        # Compute loss: if DDQN is selected, then compute losses according to the DDQN update procedure
        mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, DDQN=network_type)
        # Optimize the model according to the difference in policy and target net outputs according to the batch sampled
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor, 
         DDQN:bool=False)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
        DDQN: boolean stating whether DDQN or just DQN is run
    
    Returns:
        Float scalar tensor with loss value
    """
    if DDQN:
        q_max_vals = policy_dqn(next_states).max(1).indices.reshape([-1,1])
        bellman_targets = (~dones).reshape(-1)*target_dqn(next_states).gather(1, q_max_vals).reshape(-1) + rewards.reshape(-1)
    else:
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1) 
                                                                 # target net uses bellmann 1-step return
    
    q_values = policy_dqn(states).gather(1, actions).reshape(-1) # the policy net as the baseline
    return ((q_values - bellman_targets)**2).mean()              # updating the target net

def compute_discounted_returns(states:list[torch.Tensor], rewards:list[torch.Tensor], discount:float)->torch.Tensor:
    """ Compute discounted returns of an episode.
    
    Args:
        states: a list of episode-length length where each item indicates a state of an episode
        rewards: a list of episode-length length where each item indicates a reward of an episode
        discount: an episodic discount term
    
    Returns:
        A torch.Tensor of discounted episodic returns for a completed episode.
    """
    discounted_rewards = []

    for t in range(len(states)):
        G, power = 0, 0
        
        for reward in rewards[t:]:
            G += discount**power*reward
            power += 1

        discounted_rewards.append(G)

    discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32)
    return (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards)) # normalises returns
