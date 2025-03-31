import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable

class OrthoInit:
    """Orthogonal initialization for PyTorch layers"""
    
    @staticmethod
    def init_weights(layer, gain=1.0):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
        return layer

class SplitLayer(nn.Module):
    """Layer that splits into multiple output heads, similar to Julia's Split"""
    
    def __init__(self, paths):
        super(SplitLayer, self).__init__()
        self.paths = nn.ModuleList(paths)
    
    def forward(self, x):
        return [path(x) for path in self.paths]

class ActorNetwork(nn.Module):
    """Actor network for the PPO algorithm"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorNetwork, self).__init__()
        
        self.shared_layers = nn.Sequential(
            OrthoInit.init_weights(nn.Linear(state_size, hidden_size)),
            nn.ReLU(),
            OrthoInit.init_weights(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            OrthoInit.init_weights(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )
        
        self.mu_head = OrthoInit.init_weights(nn.Linear(hidden_size, action_size), gain=0.01)
        self.sigma_head = nn.Sequential(
            OrthoInit.init_weights(nn.Linear(hidden_size, action_size), gain=0.01),
            nn.Softplus()
        )
    
    def forward(self, state):
        features = self.shared_layers(state)
        mu = self.mu_head(features)
        sigma = self.sigma_head(features)
        return mu, sigma

class CriticNetwork(nn.Module):
    """Critic network for the PPO algorithm"""
    
    def __init__(self, state_size, hidden_size=64):
        super(CriticNetwork, self).__init__()
        
        self.model = nn.Sequential(
            OrthoInit.init_weights(nn.Linear(state_size, hidden_size)),
            nn.ReLU(),
            OrthoInit.init_weights(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            OrthoInit.init_weights(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            OrthoInit.init_weights(nn.Linear(hidden_size, 1))
        )
    
    def forward(self, state):
        return self.model(state)

class PPOPolicy:
    """
    Proximal Policy Optimization (PPO) agent
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        actor_optimizer: optim.Optimizer = None,
        critic_optimizer: optim.Optimizer = None,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        clip_range: float = 0.2,
        entropy_weight: float = 0.01,
        n_steps: int = 128,
        n_actors: int = 8,
        n_epochs: int = 4,
        batch_size: int = 64,
        target_function: Callable = None,
        hidden_size: int = 64,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize a PPO agent
        
        Args:
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
            actor_optimizer: Optimizer for the actor network
            critic_optimizer: Optimizer for the critic network
            gamma: Discount factor
            lambda_: GAE parameter
            clip_range: PPO clip range
            entropy_weight: Weight for the entropy bonus
            n_steps: Number of steps to collect per iteration
            n_actors: Number of parallel actors
            n_epochs: Number of epochs to train on each batch
            batch_size: Batch size for training
            target_function: Function to compute the target value
            hidden_size: Number of hidden units in the networks
            device: Device to use for computation
        """
        self.device = device
        
        # Create networks
        self.actor = ActorNetwork(state_size, action_size, hidden_size).to(device)
        self.critic = CriticNetwork(state_size, hidden_size).to(device)
        
        # Create optimizers if not provided
        if actor_optimizer is None:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        else:
            self.actor_optimizer = actor_optimizer
            
        if critic_optimizer is None:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        else:
            self.critic_optimizer = critic_optimizer
        
        # Set hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_range = clip_range
        self.entropy_weight = entropy_weight
        self.n_actors = n_actors
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_function = target_function
        
        # Tracking variables
        self.loss_actor = []
        self.loss_critic = []
        self.loss_entropy = []
    
    def to(self, device):
        """Move the policy to the specified device"""
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.device = device
        return self

    def cpu(self):
        """Move the policy to the CPU"""
        return self.to(torch.device('cpu'))
    
    def cuda(self):
        """Move the policy to the GPU if available"""
        if torch.cuda.is_available():
            return self.to(torch.device('cuda'))
        return self
    
    def __call__(self, states, is_sampling=True):
        """
        Compute actions for given states
        
        Args:
            states: Batch of states
            is_sampling: Whether to sample actions or return the mean
            
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of the actions
        """
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            mu, sigma = self.actor(states)
            
            if is_sampling:
                # Sample actions from the Gaussian distribution
                normal = torch.distributions.Normal(mu, sigma)
                actions = normal.sample()
            else:
                # Use the mean for deterministic actions
                actions = mu
                
        return actions.cpu().numpy() 