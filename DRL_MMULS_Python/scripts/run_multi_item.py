import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.ppo import PPOPolicy, train_ppo, TD1_target, TDL_target
from src.testbed import MultiItemInventoryEnv, TestEnvironment, EntropyAnnealing, Hook
from src.testbed.single_item import test_agent

def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent on a multi-item inventory environment')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n-actors', type=int, default=8, help='Number of actors for parallel training')
    parser.add_argument('--n-steps', type=int, default=30, help='Number of steps per actor per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lambda-gae', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--entropy-weight', type=float, default=0.01, help='Entropy weight')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num-items', type=int, default=2, help='Number of items in inventory')
    parser.add_argument('--test-freq', type=int, default=5, help='Test frequency')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = MultiItemInventoryEnv(num_items=args.num_items, num_periods=args.n_steps)
    test_env = MultiItemInventoryEnv(num_items=args.num_items, num_periods=args.n_steps)
    
    # Create agent
    agent = PPOPolicy(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        actor_optimizer=torch.optim.Adam([], lr=args.lr_actor),
        critic_optimizer=torch.optim.Adam([], lr=args.lr_critic),
        gamma=args.gamma,
        lambda_=args.lambda_gae,
        clip_range=args.clip_range,
        entropy_weight=args.entropy_weight,
        n_steps=args.n_steps,
        n_actors=args.n_actors,
        n_epochs=4,
        batch_size=args.batch_size,
        target_function=TDL_target,
        hidden_size=args.hidden_size,
        device=device
    )
    
    # Create hooks
    test_hook = TestEnvironment(test_env, test_frequency=args.test_freq)
    entropy_annealing = EntropyAnnealing(target_weight=0.001, num_iterations=args.epochs)
    hooks = Hook([test_hook, entropy_annealing])
    
    # Train agent
    agent = train_ppo(agent, env, n_iterations=args.epochs, hook=hooks)
    
    # Test agent
    mean_return, std_return = test_agent(agent, test_env, n_sims=100)
    print(f"Test performance: {mean_return:.4f} ± {std_return:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot([i * args.test_freq for i in range(len(test_hook.returns))], 
             [ret[0] for ret in test_hook.returns], 'b-', label='Mean return')
    plt.fill_between([i * args.test_freq for i in range(len(test_hook.returns))],
                    [ret[0] - ret[1] for ret in test_hook.returns],
                    [ret[0] + ret[1] for ret in test_hook.returns],
                    alpha=0.2, color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    plt.title(f'PPO Performance on Multi-Item Inventory Environment ({args.num_items} items)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs('../data', exist_ok=True)
    plt.savefig(f'../data/ppo_performance_multi_{args.num_items}.png')
    plt.show()

if __name__ == '__main__':
    main() 