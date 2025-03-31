import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy

class OrderCost:
    """Fixed plus variable order cost structure"""
    def __init__(self, K, c):
        """
        Initialize order cost
        
        Args:
            K (float): Fixed ordering cost
            c (float): Variable ordering cost
        """
        self.K = K
        self.c = c
    
    def __call__(self, quantity):
        """
        Calculate order cost
        
        Args:
            quantity (float): Order quantity
            
        Returns:
            cost (float): Order cost
        """
        if quantity > 0:
            return self.K + self.c * quantity
        return 0.0

class HoldingCost:
    """Linear holding cost structure"""
    def __init__(self, h):
        """
        Initialize holding cost
        
        Args:
            h (float): Holding cost per unit
        """
        self.h = h
    
    def __call__(self, quantity):
        """
        Calculate holding cost
        
        Args:
            quantity (float): Inventory level
            
        Returns:
            cost (float): Holding cost
        """
        return max(0, self.h * quantity)

class PenaltyCost:
    """Linear penalty cost structure"""
    def __init__(self, p):
        """
        Initialize penalty cost
        
        Args:
            p (float): Penalty cost per unit
        """
        self.p = p
    
    def __call__(self, quantity):
        """
        Calculate penalty cost
        
        Args:
            quantity (float): Backlog level
            
        Returns:
            cost (float): Penalty cost
        """
        return max(0, self.p * (-quantity))

class Source:
    """Supply source for inventory"""
    def __init__(self, order_cost, leadtime=0):
        """
        Initialize source
        
        Args:
            order_cost (OrderCost): Order cost structure
            leadtime (int): Leadtime in periods
        """
        self.order_cost = order_cost
        self.leadtime = leadtime
        self.pipeline = [0.0] * leadtime

class Item:
    """Single item in inventory system"""
    def __init__(self, sources, holding_cost, penalty_cost):
        """
        Initialize item
        
        Args:
            sources (list): List of sources
            holding_cost (HoldingCost): Holding cost structure
            penalty_cost (PenaltyCost): Penalty cost structure
        """
        self.sources = sources
        self.holding_cost = holding_cost
        self.penalty_cost = penalty_cost
        self.inventory_level = 0.0

class SingleItemInventoryEnv(gym.Env):
    """
    Single-item inventory environment that follows gym interface
    
    This environment models a single-item inventory system with:
    - Fixed plus variable ordering costs
    - Linear holding costs
    - Linear penalty costs
    - Stochastic demand
    - Multiple supply sources with different leadtimes
    """
    
    def __init__(self, num_periods=30, initial_inventory=0, demand_mean=10, demand_std=2):
        """
        Initialize environment
        
        Args:
            num_periods (int): Number of periods in horizon
            initial_inventory (float): Initial inventory level
            demand_mean (float): Mean of demand distribution
            demand_std (float): Standard deviation of demand distribution
        """
        super(SingleItemInventoryEnv, self).__init__()
        
        # Parameters
        self.num_periods = num_periods
        self.initial_inventory = initial_inventory
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        
        # Create supply source with fixed plus variable cost
        source = Source(OrderCost(K=64.0, c=1.0), leadtime=1)
        
        # Create item with holding and penalty costs
        self.bom = [Item([source], HoldingCost(h=1.0), PenaltyCost(p=9.0))]
        
        # State: inventory level + pipeline + periods remaining
        state_size = 1 + sum(len(src.pipeline) for src in self.bom[0].sources) + 1
        
        # Action: (s,S) for each source
        action_size = 2 * len(self.bom[0].sources)
        
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
        
        # Reset item
        self.bom[0].inventory_level = self.initial_inventory
        
        # Reset pipelines
        for source in self.bom[0].sources:
            source.pipeline = [0.0] * source.leadtime
        
        # Reset period counter
        self.current_period = 0
        
        # Reset cost
        self.cost = 0.0
        self.reward = 0.0
        
        # Generate demands for the entire horizon
        np.random.seed(seed)
        self.demands = np.random.normal(self.demand_mean, self.demand_std, self.num_periods)
        self.demands = np.maximum(self.demands, 0.0)  # Ensure non-negative demand
        
        return self.get_state(), {}
    
    def get_state(self):
        """
        Get current state
        
        Returns:
            state (np.array): Current state
        """
        state = [self.bom[0].inventory_level]
        
        # Add pipeline inventory
        for source in self.bom[0].sources:
            state.extend(source.pipeline)
        
        # Add remaining periods
        state.append(self.num_periods - self.current_period)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (np.array): Action to take [(s,S) for each source]
            
        Returns:
            state (np.array): Next state
            reward (float): Reward
            terminated (bool): Whether episode is terminated
            truncated (bool): Whether episode is truncated
            info (dict): Additional information
        """
        # Initialize cost for this period
        period_cost = 0.0
        
        # Process orders
        for i, source in enumerate(self.bom[0].sources):
            # Extract s and S values for this source
            s = action[2*i]
            S = action[2*i + 1]
            
            # Ensure S >= s
            S = max(S, s)
            
            # Check if inventory level is below reorder point
            if self.bom[0].inventory_level <= s:
                # Calculate order quantity
                order_quantity = S - self.bom[0].inventory_level
                
                # Add order cost
                period_cost += source.order_cost(order_quantity)
                
                # Add to pipeline
                if source.leadtime == 0:
                    self.bom[0].inventory_level += order_quantity
                else:
                    source.pipeline[0] += order_quantity
        
        # Process pipeline inventory
        for source in self.bom[0].sources:
            if source.leadtime > 0:
                # Add last pipeline element to inventory level
                self.bom[0].inventory_level += source.pipeline[-1]
                
                # Shift pipeline
                for i in range(source.leadtime - 1, 0, -1):
                    source.pipeline[i] = source.pipeline[i-1]
                source.pipeline[0] = 0.0
        
        # Process demand
        demand = self.demands[self.current_period]
        self.bom[0].inventory_level -= demand
        
        # Calculate holding and penalty costs
        period_cost += self.bom[0].holding_cost(self.bom[0].inventory_level)
        period_cost += self.bom[0].penalty_cost(self.bom[0].inventory_level)
        
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
        print(f"Inventory Level: {self.bom[0].inventory_level}")
        print(f"Pipeline: {[src.pipeline for src in self.bom[0].sources]}")
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