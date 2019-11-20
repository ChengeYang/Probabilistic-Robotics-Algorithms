#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Velocity motion model for 2D differential drive robot:
    Robot state: [x, y, θ]
    Control: [u, w].

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np


class MotionModel():
    def __init__(self, R, motion_noise):
        '''
        Input:
            R: Measurement covariance matrix.
               Dimension: [3, 3].
            motion_noise: [noise_x, noise_y, noise_theta, noise_v, noise_w]
                          (in meters / rad).
        '''
        self.R = R
        self.R_inverse = np.linalg.inv(self.R)
        self.motion_noise = motion_noise

    def initialize_particle(self, particle):
        '''
        Add motion noise to the robot state in the given particle object.

        Input:
            particle: Particle() object which has been initialized by first
                      ground truth data.
        Output:
            None.
        '''
        # Apply Gaussian noise to the robot state
        particle.x = np.random.normal(particle.x, self.motion_noise[0])
        particle.y = np.random.normal(particle.y, self.motion_noise[1])
        particle.theta = np.random.normal(particle.theta, self.motion_noise[2])

    def motion_update(self, particle, control):
        '''
        Conduct motion update for a given particle from current state X_t-1 and
        control U_t.

        Motion Model (simplified):
        State: [x, y, θ]
        Control: [v, w]
        [x_t, y_t, θ_t] = g(u_t, x_t-1)
        x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        θ_t  =  θ_t-1 + w * delta_t

        Input:
            particle: Particle() object to be updated.
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        delta_t = control[0] - particle.timestamp

        # Compute updated [timestamp, x, y, theta]
        particle.timestamp = control[0]
        particle.x += control[2] * np.cos(particle.theta) * delta_t
        particle.y += control[2] * np.sin(particle.theta) * delta_t
        particle.theta += control[3] * delta_t

        # Limit θ within [-pi, pi]
        if (particle.theta > np.pi):
            particle.theta -= 2 * np.pi
        elif (particle.theta < -np.pi):
            particle.theta += 2 * np.pi

    def sample_motion_model(self, particle, control):
        '''
        Implementation for Fast SLAM 1.0.
        Sample next state X_t from current state X_t-1 and control U_t with
        added motion noise.

        Input:
            particle: Particle() object to be updated.
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        # Apply Gaussian noise to control input
        v = np.random.normal(control[2], self.motion_noise[3])
        w = np.random.normal(control[3], self.motion_noise[4])
        control_noisy = np.array([control[0], control[1], v, w])

        # Motion updated
        self.motion_update(particle, control_noisy)


if __name__ == '__main__':
    pass
