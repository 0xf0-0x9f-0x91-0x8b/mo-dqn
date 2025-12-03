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

from learners.mo_dqn.mo_dqn_policy import MODQN
import gymnasium as gym
import mo_gymnasium as mo_gym

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
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100000,    # number of update steps
            gamma=0.9          # multiply LR by this factor
        )

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

        self.logs = {
            "Episode Return": [],
            "Multi-Objective Returns": [],
            "Loss per Update": [],
            "Average Loss per Episode": []
        }

    def train(self, num_episodes: int, ep_len: int = 200, warmup_episodes: int = 1000, warmup_ep_len: int = 100,
                epsilon_start: float = 1.0, epsilon_end: float = 0.03, log_freq: int = 100) -> None:
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
            rollout_stats = self.replay_buffer.collect_rollout(self.env, self.policy, num_steps=warmup_ep_len, epsilon=1.0)
        print('Buffer filled')
        print("-" * 60)

        # Main training loop
        print('Beginning main training loop')
        for episode in range(num_episodes):
            self.episode_count += 1

            # Collect rollout
            rollout_stats = self.replay_buffer.collect_rollout(self.env, self.policy, num_steps=ep_len, epsilon=epsilon)

            # Log 
            self.logs["Episode Return"].append(rollout_stats['episode_return'])
            if rollout_stats['mo_return'] is not None:
                self.logs["Multi-Objective Returns"].append(rollout_stats['mo_return'].tolist())

            # Perform updates
            episode_losses = []
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.updates_per_episode):
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.update(batch)
                    episode_losses.append(loss)

                    # soft update for target network
                    tau = 0.005
                    for param, target_param in zip(self.policy.model.parameters(),
                                                self.target_policy.model.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    self.logs["Loss per Update"].append(loss)
                    self.update_count += 1

            if episode_losses:
                avg_loss = np.mean(episode_losses)
                self.logs["Average Loss per Episode"].append(avg_loss)

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
        self.save_checkpoint(num_episodes, final=True)


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
        self.scheduler.step()

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
