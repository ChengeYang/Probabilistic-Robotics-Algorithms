#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Run Fast SLAM 1.0 on the UTIAS Multi-Robot Cooperative Localization and Mapping
Dataset.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import MotionModel
from lib import MeasurementModel
from src.Fast_SLAM_1_known_correspondences import FastSLAM1


if __name__ == '__main__':
    # Dataset info
    dataset = "../0.Dataset1"
    start_frame = 800
    end_frame = 3200

    # Initialize Motion Model object
    # Motion noise (in meters / rad)
    # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    # Fisrt three are used for initializing particles
    # Last two are used for motion update
    motion_noise = np.array([0.0, 0.0, 0.0, 0.1, 0.15])
    motion_model = MotionModel(motion_noise)

    # Initialize Measurement Model object
    # Measurement covariance matrix
    Q = np.diagflat(np.array([0.05, 0.02])) ** 2
    measurement_model = MeasurementModel(Q)

    # Initialize SLAM algorithm
    # Number of particles
    N_particles = 200
    fast_slam = FastSLAM1(motion_model, measurement_model)
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
        if (len(fast_slam.states) % 20 == 0):
            fast_slam.plot_data()
    plt.show()
