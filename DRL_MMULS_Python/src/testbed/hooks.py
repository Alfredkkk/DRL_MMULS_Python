import torch
import numpy as np
from copy import deepcopy
from typing import List, Callable

class Hook:
    """
    A hook for the PPO training loop that can combine multiple hooks
    """
    def __init__(self, hooks: List[Callable] = None):
        self.hooks = hooks or []
    
    def __call__(self, agent, env):
        for hook in self.hooks:
            hook(agent, env)

class EntropyAnnealing:
    """
    A hook to anneal the entropy weight during training
    """
    def __init__(self, target_weight: float, num_iterations: int):
        """
        Initialize an entropy annealing hook
        
        Args:
            target_weight: Target entropy weight
            num_iterations: Number of iterations to anneal over
        """
        self.target_weight = target_weight
        self.num_iterations = num_iterations
        self.current_iteration = 0
        self.initial_weight = None
    
    def __call__(self, agent, env):
        """
        Anneal the entropy weight
        
        Args:
            agent: PPO agent
            env: Environment
        """
        if self.initial_weight is None:
            self.initial_weight = agent.entropy_weight
        
        self.current_iteration += 1
        
        if self.current_iteration <= self.num_iterations:
            # Linear annealing
            progress = self.current_iteration / self.num_iterations
            agent.entropy_weight = self.initial_weight + progress * (self.target_weight - self.initial_weight)

class TestEnvironment:
    """
    A hook to test the agent's performance during training
    """
    def __init__(self, test_env, test_frequency: int = 1, num_episodes: int = 10):
        """
        Initialize a test environment hook
        
        Args:
            test_env: Environment to test on
            test_frequency: Frequency of testing
            num_episodes: Number of episodes to test for
        """
        self.test_env = test_env
        self.test_frequency = test_frequency
        self.num_episodes = num_episodes
        self.iteration = 0
        self.returns = []
    
    def __call__(self, agent, train_env):
        """
        Test the agent's performance
        
        Args:
            agent: PPO agent
            train_env: Training environment
        """
        self.iteration += 1
        
        if self.iteration % self.test_frequency == 0:
            returns = self._evaluate(agent)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            self.returns.append((mean_return, std_return))
            
            print(f"\nTest performance: {mean_return:.4f} ± {std_return:.4f}")
    
    def _evaluate(self, agent):
        """
        Evaluate the agent's performance
        
        Args:
            agent: PPO agent
            
        Returns:
            episode_returns: List of episode returns
        """
        device = agent.device
        episode_returns = []
        
        # Create test environments
        envs = [deepcopy(self.test_env) for _ in range(self.num_episodes)]
        
        # Initialize states
        states = np.array([env.reset()[0] for env in envs])
        dones = np.array([False for _ in range(self.num_episodes)])
        
        # Initialize returns
        returns = np.zeros(self.num_episodes)
        
        # Run episodes
        while not np.all(dones):
            # Convert states to tensor
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            
            # Get actions
            with torch.no_grad():
                actions = agent(states_tensor, is_sampling=False)
            
            # Execute actions
            for i, (env, action, done) in enumerate(zip(envs, actions, dones)):
                if not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Update states and dones
                    states[i] = next_state
                    dones[i] = done
                    
                    # Update returns
                    returns[i] += reward
            
        return returns

class Kscheduler:
    """
    A hook to schedule the order cost parameter K during training
    Similar to the Julia version
    """
    def __init__(self, k_target: float, range_start: int, range_end: int):
        """
        Initialize a K scheduler hook
        
        Args:
            k_target: Target K value
            range_start: Iteration to start increasing K
            range_end: Iteration to stop increasing K
        """
        self.n = 0
        self.k_target = k_target
        self.range = range(range_start, range_end + 1)
    
    def __call__(self, agent, env):
        """
        Update the order cost parameter K
        
        Args:
            agent: PPO agent
            env: Environment
        """
        self.n += 1
        
        if self.n in self.range:
            # Increment K
            env.bom[0].sources[0].order_cost.K += self.k_target / len(self.range) 