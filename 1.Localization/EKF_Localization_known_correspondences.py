#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of EKF Localization with known correspondences.
See Probabilistic Robotics:
    1. Page 204, Table 7.2 for full algorithm.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt


class ExtendedKalmanFilter():
    def __init__(self, dataset, end_frame, R, Q):
        self.load_data(dataset, end_frame)
        self.initialization(R, Q)
        for data in self.data:
            if (data[1] == -1):
                self.motion_update(data)
            else:
                self.measurement_update(data)
        self.plot_data()

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
        odom_data = np.insert(self.odometry_data, 1, -1, axis = 1)
        self.data = np.concatenate((odom_data, self.measurement_data), axis = 0)
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

        # Combine barcode Subject# with landmark Subject# to create lookup-table
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        self.landmark_locations = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_locations[self.barcodes_data[i][1]] = self.landmark_groundtruth_data[i - 5][1:]

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (1 - 15 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_indexes[self.barcodes_data[i][1]] = i - 4

    def initialization(self, R, Q):
        # Initial state
        self.states = np.array([self.groundtruth_data[0]])
        self.last_timestamp = self.states[-1][0]
        # Choose very small process covariance because we are using the ground truth data for initial location
        self.sigma = np.diagflat([1e-10, 1e-10, 1e-10])
        # States with measurement update
        self.states_measurement = []

        # State covariance matrix
        self.R = R
        # Measurement covariance matrix
        self.Q = Q

    def motion_update(self, control):
        # ------------------ Step 1: Mean update ---------------------#
        # State: [x, y, θ]
        # Control: [v, w]
        # State-transition function (simplified):
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t
        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if (delta_t < 0.001):
            return
        # Compute updated [x, y, theta]
        x_t = self.states[-1][1] + control[2] * np.cos(self.states[-1][3]) * delta_t
        y_t = self.states[-1][2] + control[2] * np.sin(self.states[-1][3]) * delta_t
        theta_t = self.states[-1][3] + control[3] * delta_t
        # Limit θ within [-pi, pi]
        if (theta_t > np.pi):
            theta_t -= 2 * np.pi
        elif (theta_t < -np.pi):
            theta_t += 2 * np.pi
        self.last_timestamp = control[0]
        self.states = np.append(self.states, np.array([[control[0], x_t, y_t, theta_t]]), axis = 0)

        # ------ Step 2: Linearize state-transition by Jacobian ------#
        # Jacobian: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * delta_t * sinθ_t-1
        #   G  =  0  1   v * delta_t * cosθ_t-1
        #         0  0             1
        G_1 = np.array([1, 0, - control[2] * delta_t * np.sin(self.states[-1][3])])
        G_2 = np.array([0, 1, control[2] * delta_t * np.cos(self.states[-1][3])])
        G_3 = np.array([0, 0, 1])
        self.G = np.array([G_1, G_2, G_3])

        # ---------------- Step 3: Covariance update ------------------#
        self.sigma = self.G.dot(self.sigma).dot(self.G.T) + self.R

    def measurement_update(self, measurement):
        # Continue if landmark is not found in self.landmark_locations
        if not measurement[1] in self.landmark_locations:
            return

        # ---------------- Step 1: Measurement update -----------------#
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        x_l = self.landmark_locations[measurement[1]][0]
        y_l = self.landmark_locations[measurement[1]][1]
        x_t = self.states[-1][1]
        y_t = self.states[-1][2]
        theta_t = self.states[-1][3]
        q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(y_l - y_t, x_l - x_t) - theta_t

        # -------- Step 2: Linearize Measurement by Jacobian ----------#
        # Jacobian: H = d h(x_t) / d x_t
        #        -(x_l - x_t) / sqrt(q)   -(y_l - y_t) / sqrt(q)   0
        #  H  =      (y_l - y_t) / q         -(x_l - x_t) / q     -1
        #                  0                         0             0
        #  q = (x_l - x_t)^2 + (y_l - y_t)^2
        H_1 = np.array([-(x_l - x_t) / np.sqrt(q), -(y_l - y_t) / np. sqrt(q), 0])
        H_2 = np.array([(y_l - y_t) / q, -(x_l - x_t) / q, -1])
        H_3 = np.array([0, 0, 0])
        self.H = np.array([H_1, H_2, H_3])

        # ---------------- Step 3: Kalman gain update -----------------#
        S_t = self.H.dot(self.sigma).dot(self.H.T) + self.Q
        self.K = self.sigma.dot(self.H.T).dot(np.linalg.inv(S_t))

        # ------------------- Step 4: mean update ---------------------#
        difference = np.array([measurement[2] - range_expected, measurement[3] - bearing_expected, 0])
        innovation = self.K.dot(difference)
        self.last_timestamp = measurement[0]
        self.states = np.append(self.states, np.array([[self.last_timestamp, x_t + innovation[0], y_t + innovation[1], theta_t + innovation[2]]]), axis=0)
        self.states_measurement.append([x_t + innovation[0], y_t + innovation[1]])

        # ---------------- Step 5: covariance update ------------------#
        self.sigma = (np.identity(3) - self.K.dot(self.H)).dot(self.sigma)

    def plot_data(self):
        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b', label="Robot State Ground truth")

        # States
        plt.plot(self.states[:, 1], self.states[:, 2], 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'go', label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'yo', label="End point")

        # Measurement update locations
        if (len(self.states_measurement) > 0):
            self.states_measurement = np.array(self.states_measurement)
            plt.scatter(self.states_measurement[:, 0], self.states_measurement[:, 1], s=10, c='k', alpha='0.5', label="Measurement updates")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 5
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*', label='Landmark Locations')

        # plt.title("Localization with only odometry data")
        plt.title("EKF Localization with Known Correspondences")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # # Dataset 0
    # dataset = "../0.Dataset0"
    # end_frame = 10000
    # # State covariance matrix
    # R = np.diagflat(np.array([1.0, 1.0, 1.0])) ** 2
    # # Measurement covariance matrix
    # Q = np.diagflat(np.array([350, 350, 1e16])) ** 2

    # Dataset 1
    dataset = "../0.Dataset1"
    end_frame = 3200
    # State covariance matrix
    R = np.diagflat(np.array([1.0, 1.0, 10.0])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([30, 30, 1e16])) ** 2
    #
    ekf = ExtendedKalmanFilter(dataset, end_frame, R, Q)
