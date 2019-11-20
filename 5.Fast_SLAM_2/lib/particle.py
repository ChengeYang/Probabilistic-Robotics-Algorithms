#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Definition for a single particle object.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np


class Particle():
    def __init__(self):
        # Robot state: [timestamp, x, y, 0]
        self.timestamp = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Weight
        self.weight = 1.0

        # Landmark state: [x, y]
        # All landmarks' Mean and Covariance
        self.lm_mean = np.zeros((1, 2))
        self.lm_cov = np.zeros((1, 2, 2))
        self.lm_ob = np.full(1, False)

    def initialization(self, init_state, N_particles, N_landmarks):
        '''
        Input:
            init_state: initial state (from first ground truth data)
                        [timestamp_0, x_0, y_0, θ_0]
            N_particles: number of particles in Particle Filter.
            N_landmarks: number of landmarks each particle tracks.
        Output:
            None.
        '''
        # Robot state: [timestamp, x, y, 0]
        # Initialized to init_state
        self.timestamp = init_state[0]
        self.x = init_state[1]
        self.y = init_state[2]
        self.theta = init_state[3]

        # Weight
        # Initialized to be equally distributed among all particles
        self.weight = 1.0 / N_particles

        # Landmark state: [x, y]
        # All landmarks' Mean and Covariance
        # Initialized as zeros
        self.lm_mean = np.zeros((N_landmarks, 2))
        self.lm_cov = np.zeros((N_landmarks, 2, 2))

        # Table to record if each landmark has been seen or not
        # INdex [0] - [14] represent for landmark# 6 - 20
        self.lm_ob = np.full(N_landmarks, False)


if __name__ == '__main__':
    pass
