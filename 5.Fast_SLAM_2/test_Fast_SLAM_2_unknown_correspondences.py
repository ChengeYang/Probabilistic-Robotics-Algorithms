#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run Fast SLAM 2.0 on the UTIAS Multi-Robot Cooperative Localization and Mapping
Dataset.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import MotionModel
from lib import MeasurementModel
from src.Fast_SLAM_2_unknown_correspondences import FastSLAM2


if __name__ == '__main__':
    # Dataset info
    dataset = "../0.Dataset1"
    start_frame = 400
    end_frame = 3200

    # Motion covariance matrix
    R = np.diagflat(np.array([0.01, 0.01, 0.01])) ** 2
    # Measurement covariance matrix
    # Q = np.diagflat(np.array([0.02, 0.04])) ** 2
    Q = np.diagflat(np.array([0.05, 0.10])) ** 2
    # Motion noise (in meters / rad)
    # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    # Fisrt three are used for initializing particles
    # Last two are used for motion update
    motion_noise = np.array([0.0, 0.0, 0.0, 0.1, 0.15])

    # Initialize Motion Model object
    motion_model = MotionModel(R, motion_noise)

    # Initialize Measurement Model object
    measurement_model = MeasurementModel(R, Q)

    # Initialize SLAM algorithm
    # Number of particles
    N_particles = 50
    fast_slam = FastSLAM2(motion_model, measurement_model)
    fast_slam.load_data(dataset, start_frame, end_frame)
    fast_slam.initialization(N_particles)

    # Run full Fast SLAM 1.0 algorithm
    for data in fast_slam.data:
        if (data[1] == -1):
            fast_slam.robot_update(data)
        else:
            fast_slam.landmark_update(data)
        fast_slam.state_update()
        # Plot every n frames
        if (len(fast_slam.states) % 30 == 0):
            fast_slam.plot_data()
    # fast_slam.plot_data()
    plt.show()
