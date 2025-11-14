import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import mo-gymnasium


class MODQN:
    """
    Multi-Objective Deep Q-Network agent.

    Uses a utility function to scalarize multiple objectives into a single value
    for action selection.
    """

    def __init__(self, utility_fn: List[float], num_actions: int, num_obs: int):
        """
        Initialize the MODQN agent.

        Args:
            model: QNetwork instance for Q-value prediction
            utility_fn: List of weights for each objective (should sum to 1.0)
            num_actions: Number of possible actions
            num_obs: Number of observation features
        """

        
        self.model = nn.Sequential() # TODO define a Q network that outputs a matrix num_obs x num_actions
        
        self.utility_fn = utility_fn
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.num_objectives = num_objectives
        

    def predict_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for all action-objective pairs.

        Args:
            obs: Observation array of shape (num_obs,) or (batch_size, num_obs)

        Returns:
            Q-values matrix of shape (num_actions, num_objectives) or
            (batch_size, num_actions, num_objectives) where element [i, j]
            is the value of action i under objective j
        """
        value_matrix = self.model(obs)
        return value_matrix


    def get_utility(self, values: np.ndarray) -> np.ndarray:
        """
        Scalarize multi-objective values using the utility function. 

        Args:
            values: Q-values matrix from predict_values()
            
        Should linearly apply the utility_fn weights to get a utility number for each action choice.

        Returns:
            Scalarized utility values of shape (num_actions,) or (batch_size, num_actions)
        """
        pass


    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action based on the observation.

        Calls predict_values() to get Q-values, then applies utility function
        via get_utility() to scalarize, and selects the action with highest utility.

        Args:
            obs: Observation array of shape (num_obs,)
            epsilon: Probability of selecting a random action (for exploration)

        Returns:
            Selected action index
        """
        
        value_matrix = self.predict_values(obs)
        action_utilities = self.get_utility(value_matrix)
        best_action = torch.argmax(action_utilities)
        
        return best_action


class MODQNTrainer:
    """
    Trainer class for Multi-Objective DQN.

    Handles training loop, experience replay, and model updates.
    (To be implemented later)
    """

    def __init__(self, policy: MODQN, lr: float, gamma: float, batch_size: int, replay_buffer: ReplayBuffer, env):
        """
        Initialize the MODQN trainer.

        Args:
            policy: MODQN policy to train
            lr: Learning rate for optimizer
            gamma: Discount factor for future rewards
            batch_size: Batch size for training
            replay_buffer
            env: Should be a multi-objective env following the standard from mo-gymnasium. Reward is a vector of len <num_objectives>
        """
        
        self.policy = policy
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.env = env 
    
    
    def train(self, num_episodes: int, env) -> None:
        """
        Train the MODQN network.
        
        Needs to make use of the replay buffer but I have not thought much about how that should be implemented yet - Ryan
        At minimum the buffer should probably have a method to collect rollouts to refill the buffer, and a method to sample from the buffer that the train() method calls during training.

        Args:
            num_episodes: Number of episodes to train for
            env: Gym env for training
        """
        pass


    def update(self, batch: Tuple) -> float:
        """
        Perform a single update step on the model.

        Args:
            batch: Tuple of (observations, actions, rewards, next_observations, dones)
                  where rewards has shape (batch_size, num_objectives)

        Returns:
            Loss value for this update
        """
        pass
        
    def save_checkpoint(self, ):
        """
        Save current model checkpoint with good naming conventions. Call in Trainer.train()
        """
        pass


class ReplayBuffer:
    # Define replay buffer class that contains state-action-reward trajectories. See comment in Trainer.train()
    pass
    

if __name__ == "__main__":
    
    # TODO add hyperparameter sweep.
    lr = 0.001
    gamma = 0.99
    batch_size = 64
    
    policy = MODQN()
    env = # TODO. Water reservoir or resource-gathering from https://mo-gymnasium.farama.org/environments/resource-gathering/
    trainer = MODQNTrainer(policy, lr, gamma, batch_size)
    trainer.train()