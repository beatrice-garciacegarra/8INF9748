"""
    Technologies : Gymnasium, Atari
    Projet : Agent intelligent avec Gymnasium

    Date de création : 16/06/2024
    Date de modification : 25/06/2024

    Créateur : Béatrice GARCIA CEGARRA
    Cours : Atelier Pratique en IA 2
"""

import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.optim as optim
import copy


class CNNModel(th.nn.Module):
    def __init__(self, obs_shape, n_metrics):
        super().__init__()

        n_in_chan = obs_shape[0]

        self.conv = th.nn.Sequential(
            th.nn.Conv2d(n_in_chan, 32, 3, padding='same'),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, padding='same'),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, padding='same'),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 32, 3, padding='same'),
            th.nn.ReLU(),
        )

        pre_fc_shape = self.conv(th.zeros(1, *obs_shape)).shape
        pre_fc_size = np.prod([pre_fc_shape[2], pre_fc_shape[3]])

        self.dense = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(pre_fc_size, 32, bias=True),
            th.nn.ReLU(),
            th.nn.Linear(32, n_metrics, bias=True),
            th.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x


env = gym.make('ALE/Othello-v5') # , render_mode='human' ALE/TicTacToe3D-v5 CartPole-v1
env.reset(seed=42)

for act in range(env.action_space.n):
    env_copy = copy.deepcopy(env)
    state, reward, _, _, _ = env_copy.step(act)
