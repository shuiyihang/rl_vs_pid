from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)


# %%
# Policy Network
# ~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig2.png
#
# We start by building a policy that the agent will learn using REINFORCE.
# A policy is a mapping from the current environment observation to a probability distribution of the actions to be taken.
# The policy used in the tutorial is parameterized by a neural network. It consists of 2 linear layers that are shared between both the predicted mean and standard deviation.
# Further, the single individual linear layers are used to estimate the mean and the standard deviation. ``nn.Tanh`` is used as a non-linearity between the hidden layers.
# The following function estimates a mean and standard deviation of a normal distribution from which an action is sampled. Hence it is expected for the policy to learn
# appropriate weights to output means and standard deviation based on the current observation.


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


# %%
# Building an agent
# ~~~~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig3.jpeg
#
# Now that we are done building the policy, let us develop **REINFORCE** which gives life to the policy network.
# The algorithm of REINFORCE could be found above. As mentioned before, REINFORCE aims to maximize the Monte-Carlo returns.
#
# Fun Fact: REINFROCE is an acronym for " 'RE'ward 'I'ncrement 'N'on-negative 'F'actor times 'O'ffset 'R'einforcement times 'C'haracteristic 'E'ligibility
#
# Note: The choice of hyperparameters is to train a decently performing agent. No extensive hyperparameter
# tuning was done.
#


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = torch.load('module_3')
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action



def rl_test():
    env = gym.make("InvertedPendulum-v4",render_mode="human")
    obs, info = env.reset(seed=1)

    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]

    print(env.observation_space.shape)
    print(action_space_dims)

    agent = REINFORCE(obs_space_dims, action_space_dims)

    for _ in range(2000):
        action = agent.sample_action(obs)
        # print(action.shape)
        print('angle: {:.3f}°  speed:{:.3f}m/s  action: {:.3f}N'.format(obs[1] * 57.3,obs[3], action.item()))
        # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        # print(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            # obs, info = env.reset()
    env.close()


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral = 0
        self.prev_error = 0
        self.prePrevErr = 0

    def inc_update(self, setpoint, measured_value,threshold=0):
        # Calculate error
        error = setpoint - measured_value

        if abs(error) < threshold:
            return 0
        # Proportional term
        P = self.Kp * (error - self.prev_error)

        # Integral term
        I = self.Ki * error

        # Derivative term
        D = self.Kd * (error - 2*self.prev_error + self.prePrevErr)

        # Save error for next derivative calculation
        self.prePrevErr = self.prev_error
        self.prev_error = error

        # Calculate PID output
        output = P + I + D
        return output
    def pos_update(self, setpoint, measured_value):
        # Calculate error
        error = setpoint - measured_value

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative

        # Save error for next derivative calculation
        self.prev_error = error

        # Calculate PID output
        output = P + I + D
        return output



def pid_test():
    env = gym.make("InvertedPendulum-v4",render_mode="human")
    obs, info = env.reset(seed=1)

    print("init state >>> car_pos: {}  pole angle: {}  car_vel: {} pole_ang_vel: {}".format(obs[0],obs[1],obs[2],obs[3]))
    angle_Kp = 0.5
    angle_Ki = 0.3
    angle_Kd = 0.1

    speed_Kp = 0.35
    speed_Ki = 0.3
    speed_Kd = 0.15

    car_Kp = 5
    car_Ki = 0
    car_Kd = 1
    dt = 0.02  # Time step

    pid_angle = PIDController(angle_Kp, angle_Ki, angle_Kd, dt)
    pid_speed = PIDController(speed_Kp, speed_Ki, speed_Kd, dt)

    pid_car_pos = PIDController(car_Kp, car_Ki, car_Kd, dt)
    action = np.array(0,ndmin=1)
    # [小车的位置，杆的垂直角度，小车的线速度，杆子的角速度]
    for _ in range(5000):

        pid_car_out = pid_car_pos.pos_update(0,obs[0])
        pid_ang_out= pid_angle.inc_update(pid_car_out,obs[1]*57.3)
        # action = action + np.array(pid_out,ndmin=1)
        pid_out = pid_speed.inc_update(pid_ang_out,obs[3])

        action = np.array(-1*pid_out,ndmin=1)
        action = np.clip(action,-3,3)
        print('angle: {:.3f}°  pid_ang_out:{:.6f}  speed:{:.6f}m/s  action: {:.6f}N'.format(obs[1]*57.3,-pid_ang_out,obs[3],action.item()))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            # obs, info = env.reset()
    env.close()




# 倒立摆角度控制，角速度控制,小车位置控制

pid_test()














