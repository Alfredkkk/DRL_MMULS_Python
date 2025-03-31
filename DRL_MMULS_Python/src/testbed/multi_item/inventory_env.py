import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy

class MultiItemInventoryEnv(gym.Env):
    """
    Multi-item inventory environment that follows gym interface
    
    This environment models a multi-item inventory system with:
    - Fixed plus variable ordering costs
    - Linear holding costs
    - Linear penalty costs
    - Stochastic demand
    - Multiple supply sources with different leadtimes
    """
    
    def __init__(self, num_items=2, num_periods=30, initial_inventory=None, demand_mean=None, demand_std=None):
        """
        Initialize environment
        
        Args:
            num_items (int): Number of items in the system
            num_periods (int): Number of periods in horizon
            initial_inventory (list): Initial inventory levels for each item
            demand_mean (list): Mean of demand distribution for each item
            demand_std (list): Standard deviation of demand distribution for each item
        """
        super(MultiItemInventoryEnv, self).__init__()
        
        # Parameters
        self.num_items = num_items
        self.num_periods = num_periods
        
        # Set default values if not provided
        if initial_inventory is None:
            self.initial_inventory = [0.0] * num_items
        else:
            self.initial_inventory = initial_inventory
            
        if demand_mean is None:
            self.demand_mean = [10.0] * num_items
        else:
            self.demand_mean = demand_mean
            
        if demand_std is None:
            self.demand_std = [2.0] * num_items
        else:
            self.demand_std = demand_std
        
        # Create items with supply sources
        self.bom = []
        for i in range(num_items):
            # Create supply source with fixed plus variable cost
            from ..single_item.inventory_env import Source, Item, OrderCost, HoldingCost, PenaltyCost
            source = Source(OrderCost(K=64.0, c=1.0), leadtime=1)
            
            # Create item with holding and penalty costs
            item = Item([source], HoldingCost(h=1.0), PenaltyCost(p=9.0))
            item.inventory_level = self.initial_inventory[i]
            
            self.bom.append(item)
        
        # Calculate state size: inventory level + pipeline + periods remaining for each item
        state_size = sum(1 + sum(len(src.pipeline) for src in item.sources) for item in self.bom) + 1
        
        # Calculate action size: (s,S) for each source for each item
        action_size = sum(2 * len(item.sources) for item in self.bom)
        
        # Define spaces
        high_state = np.array([float('inf')] * state_size)
        low_state = np.array([-float('inf')] * state_size)
        
        high_action = np.array([float('inf')] * action_size)
        low_action = np.array([-float('inf')] * action_size)
        
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)
        
        # Initialize
        self.reset()
    
    def reset(self, seed=None):
        """
        Reset the environment
        
        Returns:
            state (np.array): Initial state
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        # Reset items
        for i, item in enumerate(self.bom):
            item.inventory_level = self.initial_inventory[i]
            
            # Reset pipelines
            for source in item.sources:
                source.pipeline = [0.0] * source.leadtime
        
        # Reset period counter
        self.current_period = 0
        
        # Reset cost
        self.cost = 0.0
        self.reward = 0.0
        
        # Generate demands for the entire horizon
        np.random.seed(seed)
        self.demands = []
        for i in range(self.num_items):
            demand = np.random.normal(self.demand_mean[i], self.demand_std[i], self.num_periods)
            demand = np.maximum(demand, 0.0)  # Ensure non-negative demand
            self.demands.append(demand)
        
        return self.get_state(), {}
    
    def get_state(self):
        """
        Get current state
        
        Returns:
            state (np.array): Current state
        """
        state = []
        
        # Add inventory level and pipeline for each item
        for item in self.bom:
            state.append(item.inventory_level)
            
            # Add pipeline inventory
            for source in item.sources:
                state.extend(source.pipeline)
        
        # Add remaining periods
        state.append(self.num_periods - self.current_period)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (np.array): Action to take [(s,S) for each source for each item]
            
        Returns:
            state (np.array): Next state
            reward (float): Reward
            terminated (bool): Whether episode is terminated
            truncated (bool): Whether episode is truncated
            info (dict): Additional information
        """
        # Initialize cost for this period
        period_cost = 0.0
        
        # Process actions for each item
        action_idx = 0
        for item_idx, item in enumerate(self.bom):
            # Process orders for each source
            for source_idx, source in enumerate(item.sources):
                # Extract s and S values for this source
                s = action[action_idx]
                S = action[action_idx + 1]
                action_idx += 2
                
                # Ensure S >= s
                S = max(S, s)
                
                # Check if inventory level is below reorder point
                if item.inventory_level <= s:
                    # Calculate order quantity
                    order_quantity = S - item.inventory_level
                    
                    # Add order cost
                    period_cost += source.order_cost(order_quantity)
                    
                    # Add to pipeline
                    if source.leadtime == 0:
                        item.inventory_level += order_quantity
                    else:
                        source.pipeline[0] += order_quantity
        
        # Process pipeline inventory for each item
        for item in self.bom:
            for source in item.sources:
                if source.leadtime > 0:
                    # Add last pipeline element to inventory level
                    item.inventory_level += source.pipeline[-1]
                    
                    # Shift pipeline
                    for i in range(source.leadtime - 1, 0, -1):
                        source.pipeline[i] = source.pipeline[i-1]
                    source.pipeline[0] = 0.0
        
        # Process demand for each item
        for item_idx, item in enumerate(self.bom):
            demand = self.demands[item_idx][self.current_period]
            item.inventory_level -= demand
            
            # Calculate holding and penalty costs
            period_cost += item.holding_cost(item.inventory_level)
            period_cost += item.penalty_cost(item.inventory_level)
        
        # Update total cost and reward
        self.cost += period_cost
        self.reward = -period_cost
        
        # Increment period counter
        self.current_period += 1
        
        # Check if episode is done
        terminated = self.current_period >= self.num_periods
        
        return self.get_state(), self.reward, terminated, False, {}
    
    def render(self):
        """Render the environment"""
        print(f"Period: {self.current_period}/{self.num_periods}")
        for i, item in enumerate(self.bom):
            print(f"Item {i}:")
            print(f"  Inventory Level: {item.inventory_level}")
            print(f"  Pipeline: {[src.pipeline for src in item.sources]}")
        print(f"Last Period Cost: {-self.reward}")
        print(f"Total Cost: {self.cost}")
    
    def __deepcopy__(self, memo):
        """Custom deepcopy to handle env copying"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy attributes
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        
        return result 