import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PPOTrajectory(Dataset):
    """
    A class to store and manage trajectories for PPO training
    Similar to the Julia version but using PyTorch tensors
    """
    def __init__(self, capacity, state_size, action_size, device=torch.device('cpu')):
        """
        Initialize a trajectory buffer
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_size (int): Dimensionality of state space
            action_size (int): Dimensionality of action space
            device (torch.device): Device to store tensors on
        """
        self.device = device
        self.capacity = capacity
        
        # Initialize tensors
        self.states = torch.zeros((capacity, state_size), device=device)
        self.actions = torch.zeros((capacity, action_size), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_size), device=device)
        self.action_log_probs = torch.zeros((capacity, action_size), device=device)
        self.advantages = torch.zeros((capacity, 1), device=device)
        self.target_values = torch.zeros((capacity, 1), device=device)
        
        # Current position in the buffer
        self.idx = {
            'state': 0,
            'action': 0,
            'reward': 0,
            'next_state': 0,
            'action_log_prob': 0,
            'advantage': 0,
            'target_value': 0
        }
        
        # Current size of buffer
        self.size = 0
    
    def push(self, data, trace_type):
        """
        Add data to the trajectory
        
        Args:
            data (torch.Tensor): Tensor to add to the buffer (batch_size, feature_dim)
            trace_type (str): Type of data ('state', 'action', etc.)
        """
        if trace_type not in self.idx:
            raise ValueError(f"Invalid trace type: {trace_type}")
        
        batch_size = data.shape[0]
        idx = self.idx[trace_type]
        
        # Transpose data to match (feature_dim, batch_size)
        if trace_type == 'state':
            self.states[idx:idx+batch_size] = data
        elif trace_type == 'action':
            self.actions[idx:idx+batch_size] = data
        elif trace_type == 'reward':
            self.rewards[idx:idx+batch_size] = data
        elif trace_type == 'next_state':
            self.next_states[idx:idx+batch_size] = data
        elif trace_type == 'action_log_prob':
            self.action_log_probs[idx:idx+batch_size] = data
        elif trace_type == 'advantage':
            self.advantages[idx:idx+batch_size] = data
        elif trace_type == 'target_value':
            self.target_values[idx:idx+batch_size] = data
        
        self.idx[trace_type] += batch_size
        self.size = max(self.size, self.idx[trace_type])
    
    def empty(self):
        """Reset the trajectory buffer"""
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.action_log_probs.zero_()
        self.advantages.zero_()
        self.target_values.zero_()
        
        for key in self.idx:
            self.idx[key] = 0
        
        self.size = 0
    
    def get_dataloader(self, batch_size, shuffle=True):
        """
        Get a DataLoader for the trajectory
        
        Args:
            batch_size (int): Batch size for the DataLoader
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: DataLoader for the trajectory
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size
    
    def __getitem__(self, idx):
        """Get a single transition from the buffer"""
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.action_log_probs[idx],
            self.advantages[idx],
            self.target_values[idx]
        )
        
    def to_device(self, device):
        """Move the trajectory to a different device"""
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.advantages = self.advantages.to(device)
        self.target_values = self.target_values.to(device)
        self.device = device 