import itertools
from datetime import datetime
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

from mo_dqn_policy import MODQN
import gymnasium as gym
import mo_gymnasium as mo_gym
import envs.dam # For MONES


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.

    Stores state-action-reward-next_state-done tuples.
    Rewards are multi-objective vectors.
    """

    def __init__(self, max_size: int = 50000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, state: np.ndarray, action: int, reward: np.ndarray,
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Multi-objective reward vector
            next_state: Next state observation
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Args: batch_size: Number of transitions to sample
        Returns: Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current length of buffer"""
        return len(self.buffer)

    def collect_rollout(self, env, policy: MODQN, num_steps: int,
                        epsilon: float = 0.1) -> Dict[str, float]:
        """
        Collect a rollout and add transitions to the buffer.

        Args:
            env: Multi-objective environment
            policy: MODQN policy for action selection
            num_steps: Maximum number of steps to collect
            epsilon: Exploration rate for epsilon-greedy

        Returns: Dictionary with rollout statistics (episode_return, episode_length)
        """
        state, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        total_mo_return = None

        for step in range(num_steps):
            action = policy.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.add(state, action, reward, next_state, done)

            # Update statistics
            if total_mo_return is None:
                total_mo_return = np.array(reward)
            else:
                total_mo_return += np.array(reward)

            # Scalarize reward for logging
            reward_scalar = np.dot(policy.utility_fn.numpy(), reward)
            episode_return += reward_scalar
            episode_length += 1

            state = next_state

            if done:
                state, _ = env.reset()
                break

        return {
            'episode_return': episode_return,
            'episode_length': episode_length,
            'mo_return': total_mo_return
        }



class MODQNTrainer:
    """
    Trainer class for Multi-Objective DQN.

    Handles training loop, experience replay, and model updates.
    (To be implemented later)
    See here for how to interact with the env: https://mo-gymnasium.farama.org/introduction/api/
    """

    def __init__(self, policy: MODQN, lr: float, gamma: float, batch_size: int,
                 replay_buffer: ReplayBuffer, env, target_update_freq: int = 500, epsilon_decay = 0.9999, updates_per_episode=1):
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
        self.target_policy = MODQN(policy.utility_fn.tolist(), policy.num_actions, policy.num_obs, policy.num_objectives, self.policy.layer_sizes) # target policy, same size as learning policy
        self.target_policy.model.load_state_dict(policy.model.state_dict())
        self.replay_buffer = replay_buffer
        self.env = env
        self.optimizer = optim.Adam(policy.model.parameters(), lr=lr)

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_decay = epsilon_decay
        self.updates_per_episode = updates_per_episode

        self.episode_count = 0
        self.update_count = 0
        self.loss_history = []

        self.checkpoint_dir = 'outputs/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"modqn_lr{lr}_g{gamma}_UPE{self.updates_per_episode}_layers{list(self.policy.layer_sizes)}_{timestamp}"
        self.log_dir = 'outputs/logs'
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, run_name))

        # Log hyperparameters to tensorboard
        self.hparam_dict = {
            'lr': lr,
            'gamma': gamma,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq,
            'epsilon_decay': self.epsilon_decay,
            'updates_per_episode': self.updates_per_episode,
            'max_buffer_size': replay_buffer.max_size,
            'utility_fn_0': policy.utility_fn[0].item(),
            'utility_fn_1': policy.utility_fn[1].item(),
            'utility_fn_2': policy.utility_fn[2].item(),
            'layer_sizes': str(policy.layer_sizes),  # Convert to string for hparams
        }
        self.metric_dict = {'final_return': 0}  # Will be updated at end of training
        #self.writer.add_hparams(hparam_dict, metric_dict)


    def train(self, num_episodes: int, warmup_episodes: int = 100,
              epsilon_start: float = 1.0, epsilon_end: float = 0.01, log_freq: int = 10) -> None:
        """
        Train the MODQN network.

        Args:
            num_episodes: Number of episodes to train for
            warmup_episodes: Number of episodes to collect before training starts
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            updates_per_episode: Number of gradient updates per episode
            log_freq: Frequency (in episodes) to log training statistics
        """
        epsilon = epsilon_start

        print("-" * 60)
        print(f"Starting MODQN training for {num_episodes} episodes...")
        print("-" * 60)


        # Warmup phase: fill replay buffer
        print("Filling buffer")
        for _ in range(warmup_episodes):
            rollout_stats = self.replay_buffer.collect_rollout(self.env, self.policy, num_steps=1000, epsilon=1.0)
        print('Buffer filled')
        print("-" * 60)

        # Main training loop
        print('Beginning main training loop')
        for episode in range(num_episodes):
            self.episode_count += 1

            # Collect rollout
            rollout_stats = self.replay_buffer.collect_rollout(self.env, self.policy, num_steps=1000, epsilon=epsilon)

            # Log to tensorboard
            self.writer.add_scalar('train/episode_return', rollout_stats['episode_return'], self.episode_count)
            self.writer.add_scalar('train/episode_length', rollout_stats['episode_length'], self.episode_count)
            self.writer.add_scalar('train/epsilon', epsilon, self.episode_count)
            self.writer.add_scalar('train/buffer_size', len(self.replay_buffer), self.episode_count)
            if rollout_stats['mo_return'] is not None:
                for obj_idx, obj_return in enumerate(rollout_stats['mo_return']):
                    self.writer.add_scalar(f'train/objective_{obj_idx}_return', obj_return, self.episode_count)

            # Perform updates
            episode_losses = []
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.updates_per_episode):
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.update(batch)
                    episode_losses.append(loss)

                    self.writer.add_scalar('train/loss_per_update', loss, self.update_count)
                    self.update_count += 1

            if episode_losses:
                avg_loss = np.mean(episode_losses)
                self.writer.add_scalar('train/avg_loss_per_episode', avg_loss, self.episode_count)

            # Update target network
            if self.update_count > 0 and self.update_count % self.target_update_freq == 0:
                self.target_policy.model.load_state_dict(self.policy.model.state_dict())
                print(f"  [Update {self.update_count}] Target network updated")

            # Decay epsilon
            #epsilon = epsilon_start - epsilon_start * (episode / num_episodes)
            epsilon = max(epsilon_end, epsilon * self.epsilon_decay)

            # Logging
            if (episode + 1) % log_freq == 0:
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Return: {rollout_stats['episode_return']:.2f} | "
                      f"Length: {rollout_stats['episode_length']} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      )
                if rollout_stats['mo_return'] is not None:
                    print(f"  MO Return: {rollout_stats['mo_return']}")

        print("-" * 60)
        print("Training complete")
        print(f"Tensorboard logs saved to: {self.writer.log_dir}")
        self.save_checkpoint(num_episodes, final=True)
        self.writer.add_hparams(self.hparam_dict, {'final_return': rollout_stats['episode_return']})


    def update(self, batch: Tuple) -> float:
        """
        Perform a single update step on the model using DQN loss.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
                  where rewards has shape (batch_size, num_objectives)

        Returns:
            Loss value for this update
        """
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)  # (batch_size, num_objectives)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.policy.model(states).reshape(-1, self.policy.num_actions, self.policy.num_objectives)

        current_q = current_q_values[torch.arange(len(actions)), actions] # Q-values for actions taken

        # Compute target Q-values using target network
        with torch.no_grad():
            # Next Q-values from target network
            next_q_values = self.target_policy.model(next_states).reshape(-1, self.policy.num_actions, self.policy.num_objectives)

            # Apply utility function to get scalarized values for action selection
            next_utilities = torch.matmul(next_q_values, self.policy.utility_fn)
            best_actions = torch.argmax(next_utilities, dim=1)

            # Get Q-values for best actions
            next_q = next_q_values[torch.arange(len(best_actions)), best_actions]

            # Compute target
            target_q = rewards + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # Compute loss
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Log Q-value stats
        if self.update_count % 10 == 0:
            with torch.no_grad():
                self.writer.add_scalar('train/mean_q_value', current_q.mean().item(), self.update_count)
                self.writer.add_scalar('train/max_q_value', current_q.max().item(), self.update_count)
                self.writer.add_scalar('train/min_q_value', current_q.min().item(), self.update_count)
                for obj_idx in range(self.policy.num_objectives):
                    self.writer.add_scalar(f'train/mean_q_objective_{obj_idx}', current_q[:, obj_idx].mean().item(), self.update_count)

        self.loss_history.append(loss.item())
        return loss.item()

    def save_checkpoint(self, episode: int, final: bool = False) -> None:
        """
        Save current model checkpoint with good naming conventions.

        Args:
            episode: Current episode number
            final: Whether this is the final checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if final:
            filename = f"modqn_final_ep{episode}_{timestamp}.pt"
        else:
            filename = f"modqn_ep{episode}_{timestamp}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy.model.state_dict(),
            'target_model_state_dict': self.target_policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'utility_fn': self.policy.utility_fn.tolist(),
            'num_actions': self.policy.num_actions,
            'num_obs': self.policy.num_obs,
            'num_objectives': self.policy.num_objectives,
            'loss_history': self.loss_history,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")


if __name__ == "__main__":

    # Configurable parameters
    config = 'resource-gathering' # Or 'mones'
    num_episodes = 50000
    hyperparameters = { # List of HP combinations to iterate through.
        'lr': [0.0005],
        'gamma': [0.99],
        'batch_size': [256, 512],
        'utility_fn': [[0.4, 0.3, 0.3]],
        'layer_sizes': [[128, 128], [128, 256, 128]],
        'epsilon_decay': [0.9997], #0.9995,
        'updates_per_episode': [4, 7],
        'max_buffer_size':[50000]
    }

    # Best run from HP sweeps so far:
    #   Utility fn (0.4, 0.3, 0.3) is good
    #   LR 0.0005, gamma 0.99,

    # Init env and set action/obs/objective sizes
    if config == 'resource-gathering':
        num_actions, num_obs, num_objectives = 4, 4, 3
        env = mo_gym.make("resource-gathering-v0") # https://mo-gymnasium.farama.org/environments/resource-gathering/
    # elif config == 'mones':
    #     num_actions, num_obs, num_objectives = , ,
    #     env = gym.make('Dam-v0')
    else:
        raise ValueError(f'Unknown config: {config}')


    # Run hyperparameter sweep over HPs set in dictionary
    combinations = [dict(zip(hyperparameters.keys(), v)) for v in itertools.product(*hyperparameters.values())]
    print(f"Running hyperparameter sweep with {len(combinations)} configurations...")
    for i, params in enumerate(combinations):
        print(f"{'=' * 60}\nConfiguration {i + 1}/{len(combinations)}\nParameters: {params}\n{'=' * 60}\n")

        # Init classes
        policy = MODQN(params['utility_fn'], num_actions, num_obs, num_objectives, params['layer_sizes'])
        replay_buffer = ReplayBuffer(max_size=params['max_buffer_size'])
        trainer = MODQNTrainer(policy, params['lr'], params['gamma'], params['batch_size'], replay_buffer, env, epsilon_decay = params['epsilon_decay'],updates_per_episode=params['updates_per_episode'])

        trainer.train(num_episodes)