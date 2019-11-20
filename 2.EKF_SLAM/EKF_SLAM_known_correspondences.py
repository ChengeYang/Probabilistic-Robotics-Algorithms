#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of EKF SLAM with known correspondences.
See Probabilistic Robotics:
    1. Page 314, Table 10.1 for full algorithm.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilterSLAM():
    def __init__(self, dataset, end_frame, R, Q):
        self.load_data(dataset, end_frame)
        self.initialization(R, Q)
        for data in self.data:
            if (data[1] == -1):
                self.motion_update(data)
            else:
                self.measurement_update(data)
            # Plot every n frames
            if (len(self.states) > 800 and len(self.states) % 10 == 0):
                self.plot_data()
        # self.plot_data()

    def load_data(self, dataset, end_frame):
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset + "/Groundtruth.dat")
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data = np.loadtxt(dataset + "/Landmark_Groundtruth.dat")
        # Measurement: [Time[s], Subject#, range[m], bearing[rad]]
        self.measurement_data = np.loadtxt(dataset + "/Measurement.dat")
        # Odometry: [Time[s], Subject#, forward_V[m/s], angular _v[rad/s]]
        self.odometry_data = np.loadtxt(dataset + "/Odometry.dat")

        # Collect all input data and sort by timestamp
        # Add subject "odom" = -1 for odometry data
        odom_data = np.insert(self.odometry_data, 1, -1, axis=1)
        self.data = np.concatenate((odom_data, self.measurement_data), axis=0)
        self.data = self.data[np.argsort(self.data[:, 0])]

        # Remove all data before the fisrt timestamp of groundtruth
        # Use first groundtruth data as the initial location of the robot
        for i in range(len(self.data)):
            if (self.data[i][0] > self.groundtruth_data[0][0]):
                break
        self.data = self.data[i:]

        # Remove all data after the specified number of frames
        self.data = self.data[:end_frame]
        cut_timestamp = self.data[end_frame - 1][0]
        # Remove all groundtruth after the corresponding timestamp
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= cut_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[:i]

        # Combine barcode Subject# with landmark Subject#
        # Lookup table to map barcode Subjec# to landmark coordinates
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        # Ground truth data is not used in EKF SLAM
        self.landmark_locations = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_locations[self.barcodes_data[i][1]] = self.landmark_groundtruth_data[i - 5][1:]

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_indexes[self.barcodes_data[i][1]] = i - 4

        # Table to record if each landmark has been seen or not
        # Element [0] is not used. [1] - [15] represent for landmark# 6 - 20
        self.landmark_observed = np.full(len(self.landmark_indexes) + 1, False)

    def initialization(self, R, Q):
        # Initial state: 3 for robot, 2 for each landmark
        self.states = np.zeros((1, 3 + 2 * len(self.landmark_indexes)))
        self.states[0][:3] = self.groundtruth_data[0][1:]
        self.last_timestamp = self.groundtruth_data[0][0]

        # EKF state covariance: (3 + 2n) x (3 + 2n)
        # For robot states, use first ground truth data as initial value
        #   - small values for top-left 3 x 3 matrix
        # For landmark states, we have no information at the beginning
        #   - large values for rest of variances (diagonal) data
        #   - small values for all covariances (off-diagonal) data
        self.sigma = 1e-6 * np.full((3 + 2 * len(self.landmark_indexes), 3 + 2 * len(self.landmark_indexes)), 1)
        for i in range(3, 3 + 2 * len(self.landmark_indexes)):
            self.sigma[i][i] = 1e6

        # State covariance matrix
        self.R = R
        # Measurement covariance matrix
        self.Q = Q

    def motion_update(self, control):
        # ------------------ Step 1: Mean update ---------------------#
        # State: [x, y, θ, x_l1, y_l1, ......, x_ln, y_ln]
        # Control: [v, w]
        # Only robot state is updated during each motion update step!
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t
        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if (delta_t < 0.001):
            return
        # Compute updated [x, y, theta]
        x_t = self.states[-1][0] + control[2] * np.cos(self.states[-1][2]) * delta_t
        y_t = self.states[-1][1] + control[2] * np.sin(self.states[-1][2]) * delta_t
        theta_t = self.states[-1][2] + control[3] * delta_t
        # Limit θ within [-pi, pi]
        if (theta_t > np.pi):
            theta_t -= 2 * np.pi
        elif (theta_t < -np.pi):
            theta_t += 2 * np.pi
        self.last_timestamp = control[0]
        # Append new state
        new_state = np.copy(self.states[-1])
        new_state[0] = x_t
        new_state[1] = y_t
        new_state[2] = theta_t
        self.states = np.append(self.states, np.array([new_state]), axis=0)

        # ------ Step 2: Linearize state-transition by Jacobian ------#
        # Jacobian of motion: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * delta_t * sinθ_t-1
        #   G  =  0  1   v * delta_t * cosθ_t-1        0
        #         0  0             1
        #
        #                      0                    I(2n x 2n)
        self.G = np.identity(3 + 2 * len(self.landmark_indexes))
        self.G[0][2] = - control[2] * delta_t * np.sin(self.states[-2][2])
        self.G[1][2] = control[2] * delta_t * np.cos(self.states[-2][2])

        # ---------------- Step 3: Covariance update ------------------#
        # sigma = G x sigma x G.T + Fx.T x R x Fx
        self.sigma = self.G.dot(self.sigma).dot(self.G.T)
        self.sigma[0][0] += self.R[0][0]
        self.sigma[1][1] += self.R[1][1]
        self.sigma[2][2] += self.R[2][2]

    def measurement_update(self, measurement):
        # Continue if landmark is not found in self.landmark_indexes
        if not measurement[1] in self.landmark_indexes:
            return

        # Get current robot state, measurement and landmark index
        x_t = self.states[-1][0]
        y_t = self.states[-1][1]
        theta_t = self.states[-1][2]
        range_t = measurement[2]
        bearing_t = measurement[3]
        landmark_idx = self.landmark_indexes[measurement[1]]

        # If this landmark has never been seen before: initialize landmark location in the state vector as the observed one
        #   x_l = x_t + range_t * cos(bearing_t + theta_t)
        #   y_l = y_t + range_t * sin(bearing_t + theta_t)
        if not self.landmark_observed[landmark_idx]:
            x_l = x_t + range_t * np.cos(bearing_t + theta_t)
            y_l = y_t + range_t * np.sin(bearing_t + theta_t)
            self.states[-1][2 * landmark_idx + 1] = x_l
            self.states[-1][2 * landmark_idx + 2] = y_l
            self.landmark_observed[landmark_idx] = True
        # Else use current value in state vector
        else:
            x_l = self.states[-1][2 * landmark_idx + 1]
            y_l = self.states[-1][2 * landmark_idx + 2]

        # ---------------- Step 1: Measurement update -----------------#
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        delta_x = x_l - x_t
        delta_y = y_l - y_t
        q = delta_x ** 2 + delta_y ** 2
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(delta_y, delta_x) - theta_t

        # ------ Step 2: Linearize Measurement Model by Jacobian ------#
        # Landmark state becomes a variable in measurement model
        # Jacobian: H = d h(x_t, x_l) / d (x_t, x_l)
        #        1 0 0  0 ...... 0   0 0   0 ...... 0
        #        0 1 0  0 ...... 0   0 0   0 ...... 0
        # F_x =  0 0 1  0 ...... 0   0 0   0 ...... 0
        #        0 0 0  0 ...... 0   1 0   0 ...... 0
        #        0 0 0  0 ...... 0   0 1   0 ...... 0
        #          (2*landmark_idx - 2)
        #          -delta_x/√q  -delta_y/√q  0  delta_x/√q  delta_y/√q
        # H_low =   delta_y/q   -delta_x/q  -1  -delta_y/q  delta_x/q
        #               0            0       0       0          0
        # H = H_low x F_x
        F_x = np.zeros((5, 3 + 2 * len(self.landmark_indexes)))
        F_x[0][0] = 1.0
        F_x[1][1] = 1.0
        F_x[2][2] = 1.0
        F_x[3][2 * landmark_idx + 1] = 1.0
        F_x[4][2 * landmark_idx + 2] = 1.0
        H_1 = np.array([-delta_x/np.sqrt(q), -delta_y/np.sqrt(q), 0, delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
        H_2 = np.array([delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q])
        H_3 = np.array([0, 0, 0, 0, 0])
        self.H = np.array([H_1, H_2, H_3]).dot(F_x)

        # ---------------- Step 3: Kalman gain update -----------------#
        S_t = self.H.dot(self.sigma).dot(self.H.T) + self.Q
        self.K = self.sigma.dot(self.H.T).dot(np.linalg.inv(S_t))

        # ------------------- Step 4: mean update ---------------------#
        difference = np.array([range_t - range_expected, bearing_t - bearing_expected, 0])
        innovation = self.K.dot(difference)
        new_state = self.states[-1] + innovation
        self.states = np.append(self.states, np.array([new_state]), axis=0)
        self.last_timestamp = measurement[0]

        # ---------------- Step 5: covariance update ------------------#
        self.sigma = (np.identity(3 + 2 * len(self.landmark_indexes)) - self.K.dot(self.H)).dot(self.sigma)

    def plot_data(self):
        # Clear all
        plt.cla()

        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b', label="Robot State Ground truth")

        # States
        plt.plot(self.states[:, 0], self.states[:, 1], 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'g8', markersize=12, label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'y8', markersize=12, label="End point")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 5
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*', label='Landmark Ground Truth')

        # Landmark estimated locations
        estimate_xs = []
        estimate_ys = []
        for i in range(1, len(self.landmark_indexes) + 1):
            if self.landmark_observed[i]:
                estimate_xs.append(self.states[-1][2 * i + 1])
                estimate_ys.append(self.states[-1][2 * i + 2])
                plt.text(estimate_xs[-1], estimate_ys[-1], str(i+5), fontsize=10)
        plt.scatter(estimate_xs, estimate_ys, s=50, c='k', marker='.', label='Landmark Estimate')

        plt.title('EKF SLAM with known correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)
        # plt.show()


if __name__ == "__main__":
    # Dataset 1
    dataset = "../0.Dataset1"
    end_frame = 3200
    # State covariance matrix
    R = np.diagflat(np.array([5.0, 5.0, 100.0])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([110.0, 110.0, 1e16])) ** 2

    ekf_slam = ExtendedKalmanFilterSLAM(dataset, end_frame, R, Q)
    plt.show()
