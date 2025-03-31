import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
from copy import deepcopy
from .trajectory import PPOTrajectory
import time

# Constants for calculations
LOG_2PI = torch.tensor(math.log(2 * math.pi), dtype=torch.float32)
ENTROPY_CONST = torch.tensor((LOG_2PI + 1) / 2, dtype=torch.float32)

class Normalizer:
    """
    Running mean and standard deviation for normalizing data
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.mean_diff = 0.0
        self.var = 0.0
    
    def update(self, x):
        """Update the running statistics with new data"""
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        batch_count = x.shape[0]
        
        self.mean, self.n, self.mean_diff = update_mean_var_count_from_moments(
            self.mean, self.var, self.n, batch_mean, batch_var, batch_count
        )
    
    def normalize(self, x):
        """Normalize the data using the running statistics"""
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std
        
    def __call__(self, x):
        """Normalize the data"""
        return self.normalize(x)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Update mean, var, count from batch moments"""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    
    return new_mean, tot_count, new_var

def normalize_tensor(x):
    """Normalize a tensor to have mean 0 and std 1"""
    std = x.std() + 1e-8
    return x / std

def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Tensor of rewards
        values: Tensor of state values
        next_values: Tensor of next state values
        dones: Tensor of done flags
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        advantages: Tensor of advantages
    """
    # Initialize advantages
    advantages = torch.zeros_like(rewards)
    
    # Initialize gae
    gae = 0
    
    # Compute advantages in reverse order
    for t in reversed(range(len(rewards))):
        # If episode is done, there is no next state
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        
        # Store advantage
        advantages[t] = gae
    
    return advantages

def TD1_target(trajectory, agent):
    """
    Compute TD(1) target values
    
    Args:
        trajectory: Trajectory object
        agent: PPO agent
        
    Returns:
        targets: Tensor of target values
    """
    return trajectory.rewards + agent.gamma * agent.critic(trajectory.next_states)

def TDL_target(trajectory, agent):
    """
    Compute TD(λ) target values using GAE
    
    Args:
        trajectory: Trajectory object
        agent: PPO agent
        
    Returns:
        targets: Tensor of target values
    """
    return trajectory.advantages + agent.critic(trajectory.states)

def normal_log_prob(mu, sigma, x):
    """
    Compute log probability of x under a normal distribution with mean mu and std sigma
    
    Args:
        mu: Mean of the normal distribution
        sigma: Standard deviation of the normal distribution
        x: Value to compute the log probability for
        
    Returns:
        log_prob: Log probability of x
    """
    return -((x - mu) / (sigma + 1e-8))**2 / 2.0 - torch.log(sigma + 1e-8) - LOG_2PI / 2.0

def normal_entropy(sigma):
    """
    Compute entropy of a normal distribution with std sigma
    
    Args:
        sigma: Standard deviation of the normal distribution
        
    Returns:
        entropy: Entropy of the distribution
    """
    return ENTROPY_CONST + torch.log(sigma + 1e-8)

def train_ppo(agent, env, n_iterations=1000, hook=None):
    """
    Train a PPO agent on an environment
    
    Args:
        agent: PPO agent to train
        env: Environment to train on
        n_iterations: Number of iterations to train for
        hook: Function to call after each iteration
        
    Returns:
        agent: Trained PPO agent
    """
    device = agent.device
    n_actors = agent.n_actors
    n_steps = agent.n_steps
    
    # Create reward normalizer
    reward_normalizer = Normalizer()
    
    # Create trajectory buffer
    trajectory = PPOTrajectory(
        n_actors * n_steps,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        device=torch.device('cpu')
    )
    
    # Create progress bar
    progress_bar = tqdm(range(n_iterations))
    
    # Start training
    for iteration in progress_bar:
        # Create environments
        envs = [deepcopy(env) for _ in range(n_actors)]
        
        # Reset environments
        states = torch.tensor(np.array([env.reset()[0] for env in envs]), dtype=torch.float32)
        
        # Initialize arrays
        all_states = torch.zeros((n_steps, n_actors, env.observation_space.shape[0]), device=torch.device('cpu'))
        all_actions = torch.zeros((n_steps, n_actors, env.action_space.shape[0]), device=torch.device('cpu'))
        all_rewards = torch.zeros((n_steps, n_actors, 1), device=torch.device('cpu'))
        all_next_states = torch.zeros((n_steps, n_actors, env.observation_space.shape[0]), device=torch.device('cpu'))
        all_dones = torch.zeros((n_steps, n_actors, 1), device=torch.device('cpu'))
        all_log_probs = torch.zeros((n_steps, n_actors, env.action_space.shape[0]), device=torch.device('cpu'))
        all_values = torch.zeros((n_steps, n_actors, 1), device=torch.device('cpu'))
        
        # Collect experience
        for t in range(n_steps):
            # Get state values
            with torch.no_grad():
                values = agent.critic(states.to(device)).cpu()
            
            # Store states and values
            all_states[t] = states
            all_values[t] = values
            
            # Get actions
            with torch.no_grad():
                mu, sigma = agent.actor(states.to(device))
                
                # Sample actions
                normal = torch.distributions.Normal(mu, sigma)
                actions = normal.sample()
                
                # Compute log probabilities
                log_probs = normal_log_prob(mu, sigma, actions)
                
                # Move actions and log_probs to CPU
                actions = actions.cpu()
                log_probs = log_probs.cpu()
            
            # Store actions and log probs
            all_actions[t] = actions
            all_log_probs[t] = log_probs
            
            # Execute actions
            next_states = []
            rewards = []
            dones = []
            
            for i, (env_i, action) in enumerate(zip(envs, actions)):
                next_state, reward, terminated, truncated, _ = env_i.step(action.numpy())
                done = terminated or truncated
                
                next_states.append(next_state)
                rewards.append([reward])
                dones.append([float(done)])
                
                # Reset if done
                if done:
                    next_state = env_i.reset()[0]
                    next_states[i] = next_state
            
            # Convert to tensors
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            dones = torch.tensor(np.array(dones), dtype=torch.float32)
            
            # Update reward normalizer
            reward_normalizer.update(rewards)
            
            # Normalize rewards
            normalized_rewards = torch.tensor(reward_normalizer(rewards.numpy()), dtype=torch.float32)
            
            # Store next_states, rewards, and dones
            all_next_states[t] = next_states
            all_rewards[t] = normalized_rewards
            all_dones[t] = dones
            
            # Update states
            states = next_states
        
        # Compute advantages
        with torch.no_grad():
            last_values = agent.critic(states.to(device)).cpu()
        
        # Reshape arrays
        states_flat = all_states.reshape(-1, env.observation_space.shape[0])
        actions_flat = all_actions.reshape(-1, env.action_space.shape[0])
        rewards_flat = all_rewards.reshape(-1, 1)
        next_states_flat = all_next_states.reshape(-1, env.observation_space.shape[0])
        dones_flat = all_dones.reshape(-1, 1)
        log_probs_flat = all_log_probs.reshape(-1, env.action_space.shape[0])
        values_flat = all_values.reshape(-1, 1)
        
        # Compute advantages using GAE
        all_values_with_last = torch.cat([all_values.reshape(-1, 1), last_values])
        advantages = compute_gae(
            rewards_flat,
            values_flat,
            all_values_with_last[1:],
            dones_flat,
            agent.gamma,
            agent.lambda_
        )
        
        # Normalize advantages
        advantages = normalize_tensor(advantages)
        
        # Compute target values
        target_values = advantages + values_flat
        
        # Push data to trajectory
        trajectory.push(states_flat, 'state')
        trajectory.push(actions_flat, 'action')
        trajectory.push(rewards_flat, 'reward')
        trajectory.push(next_states_flat, 'next_state')
        trajectory.push(log_probs_flat, 'action_log_prob')
        trajectory.push(advantages, 'advantage')
        trajectory.push(target_values, 'target_value')
        
        # Move trajectory to device
        device_trajectory = trajectory.to_device(device)
        
        # Get data loader
        dataloader = trajectory.get_dataloader(agent.batch_size, shuffle=True)
        
        # Train actor for one epoch
        for _ in range(1):
            for states, actions, rewards, next_states, old_log_probs, advantages, target_values in dataloader:
                states = states.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                advantages = advantages.to(device)
                
                # Forward pass
                mu, sigma = agent.actor(states)
                
                # Compute new log probs
                new_log_probs = normal_log_prob(mu, sigma, actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Compute actor loss
                clip_adv = torch.clamp(ratio, 1.0 - agent.clip_range, 1.0 + agent.clip_range) * advantages
                actor_loss = -torch.min(ratio * advantages, clip_adv).mean()
                
                # Compute entropy bonus
                entropy = normal_entropy(sigma).mean()
                entropy_loss = -agent.entropy_weight * entropy
                
                # Total loss
                loss = actor_loss + entropy_loss
                
                # Optimize actor
                agent.actor_optimizer.zero_grad()
                loss.backward()
                agent.actor_optimizer.step()
                
                # Store losses
                agent.loss_actor.append(actor_loss.item())
                agent.loss_entropy.append(entropy_loss.item())
        
        # Train critic for multiple epochs
        for _ in range(agent.n_epochs):
            for states, actions, rewards, next_states, old_log_probs, advantages, target_values in dataloader:
                states = states.to(device)
                target_values = target_values.to(device)
                
                # Forward pass
                values = agent.critic(states)
                
                # Compute critic loss
                critic_loss = F.mse_loss(values, target_values)
                
                # Optimize critic
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()
                
                # Store loss
                agent.loss_critic.append(torch.sqrt(critic_loss).item())
        
        # Update clip range
        agent.clip_range *= 0.25**(1/n_iterations)
        
        # Call hook if provided
        if hook is not None:
            hook(agent, env)
        
        # Empty trajectory
        trajectory.empty()
        
        # Update progress bar
        progress_bar.set_description(
            f"Iteration {iteration+1}/{n_iterations} | "
            f"Actor loss: {agent.loss_actor[-1]:.4f} | "
            f"Critic loss: {agent.loss_critic[-1]:.4f} | "
            f"Entropy loss: {agent.loss_entropy[-1]:.4f}"
        )
    
    return agent 