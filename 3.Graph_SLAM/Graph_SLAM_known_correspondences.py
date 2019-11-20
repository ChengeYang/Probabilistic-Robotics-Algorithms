#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of Graph SLAM with known correspondences.
See Probabilistic Robotics:
    1. Page 347, Table 11.1 - 11.5 for full algorithm.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt


class GraphSLAM():
    def __init__(self, dataset, start_fram, end_frame, N_iterations, R, Q):
        self.load_data(dataset, start_frame, end_frame)
        self.initialization(R, Q)
        self.plot_data()
        for i in range(N_iterations):
            self.linearize()
            self.reduce()
            self.solve()
            self.plot_data()

    def load_data(self, dataset, start_frame, end_frame):
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

        # Select data according to start_frame and end_frame
        # Fisrt frame must be control input
        while self.data[start_frame][1] != -1:
            start_frame += 1
        # Remove all data before start_frame and after the end_timestamp
        self.data = self.data[start_frame:end_frame]
        start_timestamp = self.data[0][0]
        end_timestamp = self.data[-1][0]
        # Remove all groundtruth outside the range
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= end_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[:i]
        for i in range(len(self.groundtruth_data)):
            if (self.groundtruth_data[i][0] >= start_timestamp):
                break
        self.groundtruth_data = self.groundtruth_data[i:]

        # Combine barcode Subject# with landmark Subject#
        # Lookup table to map barcode Subjec# to landmark coordinates
        # [x[m], y[m], x std-dev[m], y std-dev[m]]
        # Ground truth data is not used in SLAM
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
        # Initial robot state: 3 for robot [x, y, theta]
        # First state is obtained from ground truth
        self.states = np.array([self.groundtruth_data[0][1:]])
        self.last_timestamp = self.data[0][0]

        # Initial landmark state: 2 for each landmark [x, y]
        self.landmark_states = np.zeros(2 * len(self.landmark_indexes))

        # State covariance matrix
        self.R = R
        self.R_inverse = np.linalg.inv(self.R)
        # Measurement covariance matrix
        self.Q = Q
        self.Q_inverse = np.linalg.inv(self.Q)

        # Compute initial state estimates based on motion model only
        for data in self.data:
            if (data[1] == -1):
                next_state = np.copy(self.states[-1])
                next_state = self.motion_update(self.states[-1], data)
                self.states = np.append(self.states, np.array([next_state]), axis=0)
                self.last_timestamp = data[0]

    def linearize(self):
        # Incorporate control and measurement constraints into information matrix & vector

        # Initialize Information Matrix and Vector
        n_state = len(self.states)
        n_landmark = len(self.landmark_indexes)
        self.omega = np.zeros((3 * n_state + 2 * n_landmark, 3 * n_state + 2 * n_landmark))
        self.xi = np.zeros(3 * n_state + 2 * n_landmark)

        # Initialize fisrt state X_0 by current estimate data
        self.omega[0:3, 0:3] = 1.0 * np.identity(3)
        self.xi[0:3] = self.states[0]

        # Initialize matrix to record at which stata the landmark is observed
        self.observe = np.full((n_state, n_landmark), False)

        # Indicators
        self.state_idx = 0
        self.last_timestamp = self.data[0][0]

        # Loop for all control and measurement data
        for data in self.data:
            # Get current state
            state = self.states[self.state_idx]
            # Incorporate control constraints
            if (data[1] == -1):
                next_state = self.motion_update(state, data)
                self.motion_linearize(state, next_state, data)
                self.state_idx += 1
            # Incorporate measurement constraints
            else:
                self.measurement_update_linearize(state, data)
            # Update indicators
            self.last_timestamp = data[0]

    def motion_update(self, current_state, control):
        # Input: current robot state X_t-1: [x_t-1, y_t-1, theta_t-1]
        #        control U_t: [timestamp, -1, v_t, w_t]
        # Output: next robot state X_t: [x_t, y_t, theta_t]

        # Motion model:
        # State: [x, y, θ, x_l1, y_l1, ......, x_ln, y_ln]
        # Control: [v, w]
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t
        delta_t = control[0] - self.last_timestamp
        x_t = current_state[0] + control[2] * np.cos(current_state[2]) * delta_t
        y_t = current_state[1] + control[2] * np.sin(current_state[2]) * delta_t
        theta_t = current_state[2] + control[3] * delta_t

        # Limit θ within [-pi, pi]
        if (theta_t > np.pi):
            theta_t -= 2 * np.pi
        elif (theta_t < -np.pi):
            theta_t += 2 * np.pi

        # Return next state
        return np.array([x_t, y_t, theta_t])

    def motion_linearize(self, current_state, next_state, control):
        # Input: current robot state X_t-1: [x_t-1, y_t-1, theta_t-1]
        #        next robot state X_t: [x_t, y_t, theta_t]
        #        control U_t: [timestamp, -1, v_t, w_t]
        # Output: None (Update Information matrix and vector)

        # Jacobian of motion: G = d g(u_t, x_t-1) / d x_t-1
        #         1  0  -v * delta_t * sinθ_t-1
        #   G  =  0  1   v * delta_t * cosθ_t-1
        #         0  0             1
        delta_t = control[0] - self.last_timestamp
        G = np.identity(3)
        G[0][2] = - control[2] * delta_t * np.sin(current_state[2])
        G[1][2] = control[2] * delta_t * np.cos(current_state[2])

        # Construct extended G = [G I]
        G_extend = np.zeros((3, 6))
        G_extend[:, 0:3] = -G
        G_extend[:, 3:6] = np.identity(3)

        # Get matrix indexes for curernt and next states
        cur_idx = 3 * self.state_idx
        nxt_idx = 3 * (self.state_idx + 1)

        # Update Information matrix at X_t-1 and X_t
        update = G_extend.T.dot(self.R_inverse).dot(G_extend)
        self.omega[cur_idx:cur_idx+3, cur_idx:cur_idx+3] += update[0:3, 0:3]
        self.omega[cur_idx:cur_idx+3, nxt_idx:nxt_idx+3] += update[0:3, 3:6]
        self.omega[nxt_idx:nxt_idx+3, cur_idx:cur_idx+3] += update[3:6, 0:3]
        self.omega[nxt_idx:nxt_idx+3, nxt_idx:nxt_idx+3] += update[3:6, 3:6]

        # Update Information vector at X_t-1 and X_t
        update = G_extend.T.dot(self.R_inverse).dot((next_state - G.dot(current_state)))
        self.xi[cur_idx:cur_idx+3] += update[0:3]
        self.xi[nxt_idx:nxt_idx+3] += update[3:6]

    def measurement_update_linearize(self, current_state, measurement):
        # Input: current full state X_t: 3 for robot, 2 for each landmark
        #        control U_t: [timestamp, #landmark, v_t, w_t]
        # Output: None (Update Information matrix and vector)

        # Continue if landmark is not found in self.landmark_indexes
        if not measurement[1] in self.landmark_indexes:
            return

        # Get current robot state
        x_t = current_state[0]
        y_t = current_state[1]
        theta_t = current_state[2]

        # Get current measurement and landmark index
        range_t = measurement[2]
        bearing_t = measurement[3]
        landmark_idx = self.landmark_indexes[measurement[1]]

        # If this landmark has never been seen before: initialize landmark location in the state vector as the observed one
        #   x_l = x_t + range_t * cos(bearing_t + theta_t)
        #   y_l = y_t + range_t * sin(bearing_t + theta_t)
        if not self.landmark_observed[landmark_idx]:
            self.landmark_states[2 * landmark_idx - 2] = x_t + range_t * np.cos(bearing_t + theta_t)
            self.landmark_states[2 * landmark_idx - 1] = y_t + range_t * np.sin(bearing_t + theta_t)
            self.landmark_observed[landmark_idx] = True

        # Get current estimate of the landmark
        x_l = self.landmark_states[2 * landmark_idx - 2]
        y_l = self.landmark_states[2 * landmark_idx - 1]

        # Expected measurement given current robot and landmark states
        # Measurement model:
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t
        delta_x = x_l - x_t
        delta_y = y_l - y_t
        q = delta_x ** 2 + delta_y ** 2
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(delta_y, delta_x) - theta_t

        # Jacobian of measurement: H = d h(x_t, x_l) / d (x_t, x_l)
        #      -delta_x/√q  -delta_y/√q  0  delta_x/√q  delta_y/√q
        # H =   delta_y/q   -delta_x/q  -1  -delta_y/q  delta_x/q
        #           0            0       0       0          0
        H_1 = np.array([-delta_x/np.sqrt(q), -delta_y/np.sqrt(q), 0, delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
        H_2 = np.array([delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q])
        H_3 = np.array([0, 0, 0, 0, 0])
        H = np.array([H_1, H_2, H_3])

        # Get matrix indexes for curernt state and landmark
        s_idx = 3 * self.state_idx
        l_idx = 3 * len(self.states) + 2 * (landmark_idx - 1)

        # Update Information matrix at X_t and m_j
        update = H.T.dot(self.Q_inverse).dot(H)
        self.omega[s_idx:s_idx+3, s_idx:s_idx+3] += update[0:3, 0:3]
        self.omega[s_idx:s_idx+3, l_idx:l_idx+2] += update[0:3, 3:5]
        self.omega[l_idx:l_idx+2, s_idx:s_idx+3] += update[3:5, 0:3]
        self.omega[l_idx:l_idx+2, l_idx:l_idx+2] += update[3:5, 3:5]

        # Update Information vector at X_t and m_j
        difference = np.array([[range_t - range_expected], [bearing_t - bearing_expected], [0]])
        state_vector = np.array([[x_t], [y_t], [theta_t], [x_l], [y_l]])
        update = H.T.dot(self.Q_inverse).dot(difference + H.dot(state_vector))
        self.xi[s_idx:s_idx+3] += update.T[0, 0:3]
        self.xi[l_idx:l_idx+2] += update.T[0, 3:5]

        # Update observation matrix
        self.observe[self.state_idx, landmark_idx - 1] = True

    def reduce(self):
        # For the current dataset, there are only 15 landmarks
        # We don't need to reduce landmark dimensions
        return

    def solve(self):
        # Add infinity to Information matrix where landmark is not observeds
        for landmark in range(len(self.landmark_indexes)):
            if self.landmark_observed[landmark + 1]:
                continue

            l_idx = 3 * len(self.states) + 2 * landmark
            self.omega[l_idx:l_idx+2, l_idx:l_idx+2] = 1e16 * np.identity(2)

        # Solve for new states and covariance matrix
        # Without reducing the dimensionality of landmarks, the landmarks locations can be solved directly from Information matrix and vector
        sigma = np.linalg.inv(self.omega)
        new_states = sigma.dot(self.xi)

        # Extract landmark locations from state vector
        self.landmark_states = new_states[3*len(self.states):]

        # Get updated state vector
        new_states = new_states[0:3*len(self.states)]
        self.states = np.resize(new_states, (int(len(new_states)/3), 3))

    def plot_data(self):
        # New figure
        plt.figure()

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
                estimate_xs.append(self.landmark_states[2 * i - 2])
                estimate_ys.append(self.landmark_states[2 * i - 1])
                plt.text(estimate_xs[-1], estimate_ys[-1], str(i+5), fontsize=10)
        plt.scatter(estimate_xs, estimate_ys, s=50, c='k', marker='.', label='Landmark Estimate')

        plt.title('Graph SLAM with known correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))


if __name__ == "__main__":
    # Dataset 1
    dataset = "../0.Dataset1"
    start_frame = 800
    end_frame = 2000
    N_iterations = 1
    # State covariance matrix
    R = np.diagflat(np.array([5, 5, 20])) ** 2
    # Measurement covariance matrix
    Q = np.diagflat(np.array([100.0, 100.0, 1e16])) ** 2

    graph_slam = GraphSLAM(dataset, start_frame, end_frame, N_iterations, R, Q)
    plt.show()
