import numpy as np


class AgentRandom():

    def __init__(self, num_agent, dim_obs, dim_action):
        self.num_agent = num_agent
        self.dim_action = dim_action
        self.action_preferences = np.ones((self.num_agent, self.dim_action))  # Initialize with uniform preferences

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action(self, obs):
        # Adjust action preferences to introduce biases or preferences
        biased_action_probs = self.action_preferences + 0.5 * np.random.randn(self.num_agent, self.dim_action)

        # Ensure that action probabilities sum to 1
        normalized_action_probs = np.array([self.softmax(probs) for probs in biased_action_probs])

        actions = [np.random.choice(self.dim_action, p=action_probs) for action_probs in normalized_action_probs]
        return np.array(actions, dtype=np.int32)  # Return actions as np.int32

    def update_action_preferences(self, agent_id, action, reward):
        # Implement an update rule to adjust action preferences based on rewards
        learning_rate = 0.3  # You can adjust this value
        self.action_preferences[agent_id, action] += learning_rate * reward

        # Ensure that the updated preference is non-negative
        self.action_preferences[self.action_preferences < 0] = 0

        # Normalize preferences to ensure they are valid probabilities
        row_sums = self.action_preferences.sum(axis=1)
        self.action_preferences /= row_sums[:, np.newaxis]

    def get_action_eval(self, obs):
        # In evaluation mode, choose actions deterministically based on learned preferences
        actions = [np.argmax(self.action_preferences[i]) for i in range(self.num_agent)]
        return np.array(actions, dtype=np.int32)  # Return actions as np.int32