import numpy as np


class UnicycleDynamics:
    def __init__(self, x=5.0, y=5.0, theta=0.0, v=0.0, w=0.0, dt=0.1):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v  # Linear velocity
        self.w = w  # Angular velocity
        self.dt = dt
    
    def update(self, action):
        """
        Update the unicycle model based on given control inputs.
        :param action: Tuple (v, w) representing linear and angular velocity.
        """
        v, w = action
        self.theta += w * self.dt
        self.theta = self.theta % (2 * np.pi)  # Keep theta within [0, 2π)
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        return np.array([self.x, self.y, self.theta])

class ModifiedUnicycleDynamics: #from the paper
    def __init__(self, x=5.0, y=5.0, theta=0.0, v=0.0, w=0.0, dt=0.1, epsilon=0.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w = w
        self.dt = dt
        self.epsilon = epsilon  # Reference point offset
    
    def update(self, action):
        """
        Update the modified unicycle model using an offset reference point.
        :param action: Tuple (a_v, a_theta) representing forward velocity and steering speed.
        """
        v, w = action
        self.theta = (self.theta + w * self.dt) % (2 * np.pi)
        self.x += (np.cos(self.theta) * v - self.epsilon * np.sin(self.theta) * w) * self.dt
        self.y += (np.sin(self.theta) * v + self.epsilon * np.cos(self.theta) * w) * self.dt
        return np.array([self.x, self.y, self.theta])


    def get_robot_coordinates(self):
        """Computes the actual robot center from the shifted reference point."""
        x_center = self.x - self.epsilon * np.cos(self.theta)
        y_center = self.y - self.epsilon * np.sin(self.theta)
        return np.array([x_center, y_center, self.theta])
    

class SingleIntegratorDynamics:
    def __init__(self, x=5.0, y=5.0, dt=1):
        self.x = x
        self.y = y
        self.dt = dt

    def update(self, action):
        """
        Update the single integrator model based on given control inputs.
        :param action: Tuple (vx, vy) representing velocity in x and y directions.
        """
        vx, vy = action
        self.x += vx * self.dt
        self.y += vy * self.dt
        return np.array([self.x, self.y])