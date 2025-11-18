import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import mo_gymnasium


class MODQN:
    """
    Multi-Objective Deep Q-Network agent.

    Multi-objective Q-network that predicts Q-values for each action-objective pair, then uses a utility function to
    scalarize multiple objectives into a single value for action selection.
    """

    def __init__(self, utility_fn: List[float], num_actions: int, num_obs: int, num_objectives: int, layer_sizes: List[int]):
        """
        Initialize the MODQN agent.

        Args:
            utility_fn: List of weights for each objective (should sum to 1.0), e.g. [0.4, 0.3, 0.1] for a 3-objective problem
            num_actions: Number of possible actions
            num_obs: Number of observation features
            num_objectives: Number of objectives to optimize
            layer_sizes: List of layer sizes for the model
        """

        # Build the model dynamically based on layer_sizes
        layers = [nn.Linear(num_obs, layer_sizes[0]), nn.ReLU()]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], num_actions * num_objectives))

        self.model = nn.Sequential(*layers)

        self.model = nn.Sequential(
            nn.Linear(num_obs, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], num_actions * num_objectives)
        )
        self.layer_sizes = layer_sizes

        self.utility_fn = torch.tensor(utility_fn, dtype=torch.float32)
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.num_objectives = num_objectives


    def predict_values(self, obs: np.ndarray) -> torch.Tensor:
        """
        Predict Q-values for all action-objective pairs.

        Args:
            obs: Observation array of shape (num_obs,) or (batch_size, num_obs)

        Returns:
            Q-values matrix of shape (num_actions, num_objectives) or
            (batch_size, num_actions, num_objectives) where element [i, j]
            is the value of action i under objective j
        """

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(obs)

        if obs.dim() == 1: # Single observation
            value_matrix = output.reshape(self.num_actions, self.num_objectives)
        else: # Batch of observations
            batch_size = obs.shape[0]
            value_matrix = output.reshape(batch_size, self.num_actions, self.num_objectives)

        return value_matrix


    def get_utility(self, values: torch.Tensor) -> torch.Tensor:
        """
        Scalarize multi-objective values using the utility function.
        utility = sum(weight_i * value_i) for each action

        Args:
            values: Q-values matrix from predict_values()

        Should linearly apply the utility_fn weights to get a utility number for each action choice.

        Returns:
            utilities: Scalarized utility values of shape (num_actions,) or (batch_size, num_actions)
        """

        if values.dim() == 2: # Single observation case: (num_actions, num_objectives)
            utilities = torch.matmul(values, self.utility_fn)
        else: # Batch case: (batch_size, num_actions, num_objectives)
            utilities = torch.matmul(values, self.utility_fn)

        return utilities


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
        # Epsilon-greedy for now
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)

        value_matrix = self.predict_values(obs)
        action_utilities = self.get_utility(value_matrix)
        best_action = torch.argmax(action_utilities).item()

        return best_action