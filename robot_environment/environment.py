import numpy as np

class ContinuousRobotEnv:
    def __init__(self, n_agents, space_size=10.0):
        """
        Initialize the environment.
        
        Args:
            n_agents (int): Number of robotic agents.
            space_size (float): Size of the 2D square space (default 10.0x10.0).
        """
        self.n_agents = n_agents
        self.space_size = space_size
        self.reset()

    def reset(self):
        """
        Reset all agent positions randomly in the space.
        """
        self.positions = np.random.uniform(low=0.0, high=self.space_size, size=(self.n_agents, 2))
        return self.positions

    def step(self, actions):
        """
        Update agent positions based on provided action vectors.
        
        Args:
            actions (np.ndarray): An (n_agents, 2) array where each row is (delta_x, delta_y).
        """
        assert actions.shape == (self.n_agents, 2), "Actions must be (n_agents, 2) array"
        
        # Update positions
        self.positions += actions
        
        # Clip positions to stay inside [0, space_size] bounds
        self.positions = np.clip(self.positions, 0.0, self.space_size)
        
        return self.positions

    def get_state(self):
        """
        Get current positions of agents.
        
        Returns:
            np.ndarray: (n_agents, 2) array of positions.
        """
        return self.positions.copy()
