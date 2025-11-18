import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from mo_dqn_policy import MODQN
import gymnasium as gym
import mo_gymnasium as mo_gym
import envs.dam # For MONES


class MODQNTrainer:
    """
    Trainer class for Multi-Objective DQN.

    Handles training loop, experience replay, and model updates.
    (To be implemented later)
    See here for how to interact with the env: https://mo-gymnasium.farama.org/introduction/api/
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
    
    
    def train(self, num_episodes: int) -> None:
        """
        Train the MODQN network.
        
        Needs to make use of the replay buffer but I have not thought much about how that should be implemented yet - Ryan


        Args:
            num_episodes: Number of episodes to train for
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
    # TODO Define replay buffer class that contains state-action-reward trajectories. At minimum the buffer should
    #  probably have a method to collect rollouts to refill the buffer, and a method to sample from the buffer that
    #  the train() method calls during training.
    pass
    

if __name__ == "__main__":
    
    # Configurable parameters
    config = 'resource-gathering' # Or 'mones'
    # TODO add hyperparameter sweep.
    lr = 0.001
    gamma = 0.99
    batch_size = 64
    num_episodes = 1000
    utility_fn = [0.4, 0.3, 0.3]
    layer_sizes = [64, 64]

    # Init env and set action/obs/objective sizes
    if config == 'resource-gathering':
        num_actions, num_obs, num_objectives = 4, 4, 3
        env = mo_gym.make("resource-gathering-v0") # https://mo-gymnasium.farama.org/environments/resource-gathering/
    elif config == 'mones':
        num_actions, num_obs, num_objectives = , ,
        env = gym.make('Dam-v0')
    else:
        raise ValueError(f'Unknown config: {config}')

    # Init classes
    policy = MODQN(utility_fn, num_actions, num_obs, num_objectives, layer_sizes)
    replay_buffer = ReplayBuffer()
    trainer = MODQNTrainer(policy, lr, gamma, batch_size, replay_buffer, env)

    trainer.train(num_episodes)