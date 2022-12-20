import random
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, layer_sizes, activation='F.relu'):
        """
        DQN initialisation

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

def initialise_networks(network_layers):
    """Initialise policy and target networks.

    Args:
        network_layers (list[int]): _description_

    Returns: None
    """
    
    policy_net = DQN(network_layers)                                           # policy net of layers network_layers
    target_net = DQN(network_layers)                                           # target net of layers network_layers
    update_target(target_net, policy_net)                                      # load target_net with parameters of policy_net
    target_net.eval()                                                          # set the target net to a no-learning (eval) mode
    return policy_net, target_net

def greedy_action(dqn, state):
    """Select action according to a given DQN
    
    Args:
        dqn:DQN the DQN that selects the action
        state:torch.Tensor state at which the action is chosen

    Returns:
        Greedy action according to DQN (int)
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon, dqn, state):
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon:float parameter for epsilon-greedy action selection
        dqn:DQN the DQN that selects the action
        state:torch.Tensor state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action (int)
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon: return greedy_act
    else: return random.randint(0,num_actions-1)

def update_target(target_dqn, policy_dqn, episode=None, threshold=None):
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

def update_policy(memory, policy_net, target_net, optimizer, network_type, batch_size):
    """Update policy network parameters using a gradient descent (or related) scheme.
    Does not return anything but modifies the policy network passed as parameter

    Args:
        memory (ReplayBuffer): A Replay Buffer consisting of past episode steps as memory.
        policy_net (DQN): A DQN that 
        target_ne (DQN): _description_
        optimizer (torch.optim): _description_
        network_type (str): _description_
        batch_size (int): _description_
    
    Returns:
        pol
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

def loss(policy_dqn, target_dqn,
         states, actions,
         rewards, next_states, dones, 
         DDQN=False):
    """Calculate Bellman error loss
    
    Args:
        policy_dqn (DQN): policy DQN
        target_dqn (DQN): target DQN
        states (torch.Tensor): batched state tensor
        actions (torch.Tensor): batched action tensor
        rewards (torch.Tensor): batched rewards tensor
        next_states (torch.Tensor): batched next states tensor
        dones (torch.Tensor): batched Boolean tensor, True when episode terminates
        DDQN (bool): boolean stating whether DDQN or just DQN is run
    
    Returns:
        Float scalar tensor with loss value (torch.Tensor)
    """
    if DDQN: 
        q_max_vals = policy_dqn(next_states).max(1).indices.reshape([-1,1])
        bellman_targets = (~dones).reshape(-1)*target_dqn(next_states).gather(1, q_max_vals).reshape(-1) + rewards.reshape(-1)
    else:
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()
