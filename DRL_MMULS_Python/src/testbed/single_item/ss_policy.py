import numpy as np
import torch
from copy import deepcopy

def test_ss_policy(env, s, S, n=1000):
    """
    Test an (s,S) policy on an inventory environment
    
    Args:
        env: Inventory environment
        s: Reorder level
        S: Order-up-to level
        n: Number of simulations
    
    Returns:
        mean_return: Mean return across simulations
        std_return: Standard deviation of returns
    """
    # Create multiple environments for parallel evaluation
    envs = [deepcopy(env) for _ in range(n)]
    returns = np.zeros(n)
    
    # Run the policy on each environment
    for i, environment in enumerate(envs):
        for t in range(env.num_periods):
            environment.step([s[t], S[t]])
            returns[i] -= environment.reward
    
    return np.mean(returns), np.std(returns)

def test_agent(agent, env, n_sims=1000):
    """
    Test a PPO agent on an inventory environment
    
    Args:
        agent: PPO agent
        env: Inventory environment
        n_sims: Number of simulations
    
    Returns:
        mean_return: Mean return across simulations
        std_return: Standard deviation of returns
    """
    # Create multiple environments for parallel evaluation
    envs = [deepcopy(env) for _ in range(n_sims)]
    returns = np.zeros(n_sims)
    
    # Get device
    device = agent.device
    
    # Reset environments
    for environment in envs:
        environment.reset()
    
    # Flag for environment completion
    dones = np.zeros(n_sims, dtype=bool)
    
    # Run episodes until all are done
    while not np.all(dones):
        # Get states from all non-terminated environments
        states = []
        active_envs = []
        
        for i, env in enumerate(envs):
            if not dones[i]:
                states.append(env.get_state())
                active_envs.append((i, env))
        
        if not active_envs:
            break
        
        # Convert states to tensor
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        
        # Get actions from agent
        actions = agent(states_tensor, is_sampling=False)
        
        # Apply actions to environments
        for (i, env), action in zip(active_envs, actions):
            state, reward, done, _, _ = env.step(action)
            returns[i] += reward
            dones[i] = done
    
    return np.mean(returns), np.std(returns) 