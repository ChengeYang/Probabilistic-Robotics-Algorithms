#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of Particle Filter Localization with known correspondences.
See Probabilistic Robotics:
    1. Page 252, Table 8.2 for main algorithm.
    2. Page 124, Table 5.3 for motion model.
    3. Page 179, Table 6.4 for measurement model.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class ParticleFilter():
    def __init__(self, dataset, end_frame, num_particles, motion_noise, measurement_noise):
        self.load_data(dataset, end_frame)
        self.initialization(num_particles, motion_noise, measurement_noise)
        for data in self.data:
            if (data[1] == -1):
                self.motion_update(data)
            else:
                self.measurement_update(data)
                self.importance_sampling()
            self.state_update()
            # Plot every n frames
            if (len(self.states) > 800 and len(self.states) % 15 == 0):
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
        odom_data = np.insert(self.odometry_data, 1, -1, axis=1)
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

    def initialization(self, num_particles, motion_noise, measurement_noise):
        # Initial state: use first ground truth data
        self.states = np.array([self.groundtruth_data[0]])
        self.last_timestamp = self.states[-1][0]

        # Noise covariance
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

        # Initial particles: set with initial state mean and normalized noise
        # num_particles of [x, y, theta]
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = np.random.normal(self.states[-1][1], self.motion_noise[0], num_particles)
        self.particles[:, 1] = np.random.normal(self.states[-1][2], self.motion_noise[1], num_particles)
        self.particles[:, 2] = np.random.normal(self.states[-1][3], self.motion_noise[2], num_particles)

        # Initial weights: set with uniform weights for each particle
        self.weights = np.full(num_particles, 1.0 / num_particles)

    def motion_update(self, control):
        # Motion Model (simplified):
        # State: [x, y, θ]
        # Control: [v, w]
        # [x_t, y_t, θ_t] = g(u_t, x_t-1)
        #   x_t  =  x_t-1 + v * cosθ_t-1 * delta_t
        #   y_t  =  y_t-1 + v * sinθ_t-1 * delta_t
        #   θ_t  =  θ_t-1 + w * delta_t

        # Skip motion update if two odometry data are too close
        delta_t = control[0] - self.last_timestamp
        if (delta_t < 0.1):
            return

        for particle in self.particles:
            # Apply noise to control input
            v = np.random.normal(control[2], self.motion_noise[3], 1)
            w = np.random.normal(control[3], self.motion_noise[4], 1)

            # Compute updated [x, y, theta]
            particle[0] += v * np.cos(particle[2]) * delta_t
            particle[1] += v * np.sin(particle[2]) * delta_t
            particle[2] += w * delta_t

            # Limit θ within [-pi, pi]
            if (particle[2] > np.pi):
                particle[2] -= 2 * np.pi
            elif (particle[2] < -np.pi):
                particle[2] += 2 * np.pi

        # Update timestamp
        self.last_timestamp = control[0]

    def measurement_update(self, measurement):
        # Measurement Model:
        #   range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        #  bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t

        # Continue if landmark is not found in self.landmark_locations
        if not measurement[1] in self.landmark_locations:
            return

        # Importance factor: update weights for each particle (Table 6.4)
        x_l = self.landmark_locations[measurement[1]][0]
        y_l = self.landmark_locations[measurement[1]][1]
        for i in range(len(self.particles)):
            # Compute expected range and bearing given current pose
            x_t = self.particles[i][0]
            y_t = self.particles[i][1]
            theta_t = self.particles[i][2]
            q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
            range_expected = np.sqrt(q)
            bearing_expected = np.arctan2(y_l - y_t, x_l - x_t) - theta_t

            # Compute the probability of range and bearing differences in normal distribution with mean = 0 and sigma = measurement noise
            range_error = measurement[2] - range_expected
            bearing_error = measurement[3] - bearing_expected
            prob_range = stats.norm(0, self.measurement_noise[0]).pdf(range_error)
            prob_bearing = stats.norm(0, self.measurement_noise[1]).pdf(bearing_error)

            # Update weights
            self.weights[i] = prob_range * prob_bearing

        # Normalization
        # Avoid all-zero weights
        if (np.sum(self.weights) == 0):
            self.weights = np.ones_like(self.weights)
        self.weights /= np.sum(self.weights)

        # Update timestamp
        self.last_timestamp = measurement[0]

    def importance_sampling(self):
        # Resample according to importance weights
        new_idexes = np.random.choice(len(self.particles), len(self.particles), replace = True, p = self.weights)

        # Update new particles
        self.particles = self.particles[new_idexes]

    def state_update(self):
        # Update robot pos by mean of all particles
        state = np.mean(self.particles, axis = 0)
        self.states = np.append(self.states, np.array([[self.last_timestamp, state[0], state[1], state[2]]]), axis = 0)

    def plot_data(self):
        # Clear all
        plt.cla()

        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2], 'b', label="Robot State Ground truth")

        # States
        plt.plot(self.states[:, 1], self.states[:, 2], 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2], 'go', label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2], 'yo', label="End point")

        # Particles
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=5, c='k', alpha=0.5, label="Particles")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 5
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*', label='Landmark Locations')

        plt.title("Particle Filter Localization with Known Correspondences")
        plt.legend()

        # Dataset 0
        # plt.xlim((0.3, 3.7))
        # plt.ylim((-0.6, 2.7))
        # plt.pause(1e-16)

        # Dataset 1
        plt.xlim((-1.5, 5.0))
        plt.ylim((-6.5, 6.0))
        plt.pause(1e-16)

        # Plot at the end
        # plt.show()


if __name__ == "__main__":
    # # Dataset 0
    # dataset = "../0.Dataset0"
    # end_frame = 10000
    # # Number of particles
    # num_particles = 50
    # # Motion model noise (in meters / rad)
    # # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    # motion_noise = np.array([0.2, 0.2, 0.2, 0.1, 0.1])
    # # Measurement model noise (in meters / rad)
    # # [noise_range, noise_bearing]
    # measurement_noise = np.array([0.1, 0.2])

    # Dataset 1
    dataset = "../0.Dataset1"
    end_frame = 3200
    # Number of particles
    num_particles = 50
    # Motion model noise (in meters / rad)
    # [noise_x, noise_y, noise_theta, noise_v, noise_w]
    motion_noise = np.array([0.1, 0.1, 0.1, 0.2, 0.2])
    # Measurement model noise (in meters / rad)
    # [noise_range, noise_bearing]
    measurement_noise = np.array([0.1, 0.1])

    pf = ParticleFilter(dataset, end_frame, num_particles, motion_noise, measurement_noise)
    plt.show()
