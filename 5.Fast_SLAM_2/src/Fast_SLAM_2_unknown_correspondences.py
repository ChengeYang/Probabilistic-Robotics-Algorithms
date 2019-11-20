#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Implementation of Fast SLAM 2.0 with unknown correspondences.

In general, Fast SLAM 2.0 is more difficult in implementation.
The main difference from Fast SLAM 1.0 lies in the functions:
    1. data_association
    2. landmark_update
    3. sample_motion_measurement_model
We sample the posterior considering the effect of measurement.

See Probabilistic Robotics:
    1. Page 463, Table 13.3 for full algorithm.

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np
import matplotlib.pyplot as plt
import copy

from lib.particle import Particle


class FastSLAM2():
    def __init__(self, motion_model, measurement_model):
        '''
        Input:
            motion_model: ModelModel() object
            measurement_model: MeasurementModel() object
        '''
        self.motion_model = motion_model
        self.measurement_model = measurement_model

    def load_data(self, dataset, start_frame, end_frame):
        '''
        Load data from UTIAS Multi-Robot Cooperative Localization and Mapping
        Dataset.

        Input:
            dataset: directory of the dataset.
            start_frame: the index of first frame to run the SLAM algorithm on.
            end_frame: the index of last frame to run the SLAM algorithm on.
        Otput:
            None.
        '''
        # Loading dataset
        # Barcodes: [Subject#, Barcode#]
        self.barcodes_data = np.loadtxt(dataset + "/Barcodes.dat")
        # Ground truth: [Time[s], x[m], y[m], orientation[rad]]
        self.groundtruth_data = np.loadtxt(dataset + "/Groundtruth.dat")
        # Landmark ground truth: [Subject#, x[m], y[m]]
        self.landmark_groundtruth_data =\
            np.loadtxt(dataset + "/Landmark_Groundtruth.dat")
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
            self.landmark_locations[self.barcodes_data[i][1]] =\
                self.landmark_groundtruth_data[i - 5][1:]

        # Lookup table to map barcode Subjec# to landmark Subject#
        # Barcode 6 is the first landmark (0 - 14 for 6 - 20)
        self.landmark_indexes = {}
        for i in range(5, len(self.barcodes_data), 1):
            self.landmark_indexes[self.barcodes_data[i][1]] = i - 5

    def initialization(self, N_particles):
        '''
        Initialize robots state, landmark state and all particles.

        Input:
            N_particles: number of particles this SLAM algorithms tracks.
        Output:
            None.
        '''
        # Number of particles and landmarks
        self.N_particles = N_particles
        self.N_landmarks = len(self.landmark_indexes)

        # Robot states: [timestamp, x, y, theta]
        # First state is obtained from ground truth
        self.states = np.array([self.groundtruth_data[0]])

        # Landmark states: [x, y]
        self.landmark_states = np.zeros((self.N_landmarks, 2))

        # Table to record if each landmark has been seen or not
        # [0] - [14] represent for landmark# 6 - 20
        self.landmark_observed = np.full(self.N_landmarks, False)

        # Initial particles
        self.particles = []
        for i in range(N_particles):
            particle = Particle()
            particle.initialization(self.states[0], self.N_particles,
                                    self.N_landmarks)
            self.motion_model.initialize_particle(particle)
            self.particles.append(particle)

    def robot_update(self, control):
        '''
        Update robot pose through sampling motion model for all particles.

        Input:
            control: control input U_t.
                     [timestamp, -1, v_t, w_t]
        Output:
            None.
        '''
        for particle in self.particles:
            self.motion_model.sample_motion_model(particle, control)

    def landmark_update(self, measurement):
        '''
        Update landmark mean and covariance for all landmarks of all particles.
        Based on EKF method.

        Input:
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
        Output:
            None.
        '''
        # Return if the measured object is not a landmark (another robot)
        if not measurement[1] in self.landmark_indexes:
            return

        for particle in self.particles:
            # Trick!!!!!: if the current landmark has not been seen, return its
            # corresponding index, and initilize it at the measured location
            if not particle.lm_ob[self.landmark_indexes[measurement[1]]]:
                landmark_idx =  self.landmark_indexes[measurement[1]]
                self.measurement_model.\
                    initialize_landmark(particle, measurement,
                                        landmark_idx, 1.0/self.N_landmarks)

            # Update landmark by EKF if it has been observed before
            else:
                landmark_idx, pose_sample =\
                    self.data_association(particle, measurement)
                self.measurement_model.\
                    landmark_update(particle, measurement, landmark_idx,
                                    pose_sample)

        # Normalize all weights
        self.weights_normalization()

        # Resample all particles according to the weights
        self.importance_sampling()

    def data_association(self, particle, measurement):
        '''
        For a particle, compute likelihood of correspondence for all observed
        landmarks.
        Choose landmark according to ML (Maximum Likelihood).

        Input:
            particle: single particle to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
        Output:
            landmark_idx: index of the associated landmark.
            pose_sample: [x, t, theta]
                         Robot pose sampled from the joint posterior
                         p(X_t | X_1:t-1, U_1:t, Z_1:t, C_1:t).
        '''
        # Return if the measured object is not a landmark (another robot)
        if not measurement[1] in self.landmark_indexes:
            return

        # Compute likelihood and pose sample for all observed landmarks
        likelihood_array = np.zeros(self.N_landmarks)
        pose_sample_array = np.zeros((self.N_landmarks, 3))

        for landmark_idx in range(self.N_landmarks):
            if not particle.lm_ob[landmark_idx]:
                continue

            pose_sample, Q = self.measurement_model.\
                sample_measurement_model(particle, measurement, landmark_idx)

            likelihood = self.measurement_model.\
                compute_correspondence(particle, measurement, landmark_idx,
                                       pose_sample, Q)

            likelihood_array[landmark_idx] = likelihood
            pose_sample_array[landmark_idx] = pose_sample.T

        # Selete landmark with ML correspondence and corresponding sampled pose
        landmark_idx = np.argmax(likelihood_array)
        pose_sample = pose_sample_array[landmark_idx]

        return landmark_idx, pose_sample

    def weights_normalization(self):
        '''
        Normalize weight in all particles so that the sum = 1.

        Input:
            None.
        Output:
            None.
        '''
        # Compute sum of the weights
        sum = 0.0
        for particle in self.particles:
            sum += particle.weight

        # If sum is too small, equally assign weights to all particles
        if sum < 1e-10:
            for particle in self.particles:
                particle.weight = 1.0 / self.N_particles
            return

        for particle in self.particles:
            particle.weight /= sum

    def importance_sampling(self):
        '''
        Resample all particles through the importance factors.

        Input:
            None.
        Output:
            None.
        '''
        # Construct weights vector
        weights = []
        for particle in self.particles:
            weights.append(particle.weight)

        # Resample all particles according to importance weights
        new_indexes =\
            np.random.choice(len(self.particles), len(self.particles),
                             replace=True, p=weights)

        # Update new particles
        new_particles = []
        for index in new_indexes:
            new_particles.append(copy.deepcopy(self.particles[index]))
        self.particles = new_particles

    def state_update(self):
        '''
        Update the robot and landmark states by taking average among all
        particles.

        Input:
            None.
        Output:
            None.
        '''
        # Robot state
        timestamp = self.particles[0].timestamp
        x = 0.0
        y = 0.0
        theta = 0.0

        for particle in self.particles:
            x += particle.x
            y += particle.y
            theta += particle.theta

        x /= self.N_particles
        y /= self.N_particles
        theta /= self.N_particles

        self.states = np.append(self.states,
                                np.array([[timestamp, x, y, theta]]), axis=0)

        # Landmark state
        landmark_states = np.zeros((self.N_landmarks, 2))
        count = np.zeros(self.N_landmarks)
        self.landmark_observed = np.full(self.N_landmarks, False)

        for particle in self.particles:
            for landmark_idx in range(self.N_landmarks):
                if particle.lm_ob[landmark_idx]:
                    landmark_states[landmark_idx] +=\
                        particle.lm_mean[landmark_idx]
                    count[landmark_idx] += 1
                    self.landmark_observed[landmark_idx] = True

        for landmark_idx in range(self.N_landmarks):
            if self.landmark_observed[landmark_idx]:
                landmark_states[landmark_idx] /= count[landmark_idx]

        self.landmark_states = landmark_states

    def plot_data(self):
        '''
        Plot all data through matplotlib.
        Conduct animation as the algorithm runs.

        Input:
            None.
        Output:
            None.
        '''
        # Clear all
        plt.cla()

        # Ground truth data
        plt.plot(self.groundtruth_data[:, 1], self.groundtruth_data[:, 2],
                 'b', label="Robot State Ground truth")

        # States
        plt.plot(self.states[:, 1], self.states[:, 2],
                 'r', label="Robot State Estimate")

        # Start and end points
        plt.plot(self.groundtruth_data[0, 1], self.groundtruth_data[0, 2],
                 'g8', markersize=12, label="Start point")
        plt.plot(self.groundtruth_data[-1, 1], self.groundtruth_data[-1, 2],
                 'y8', markersize=12, label="End point")

        # Particles
        particle_xs = []
        particle_ys = []
        for particle in self.particles:
            particle_xs.append(particle.x)
            particle_ys.append(particle.y)
        plt.scatter(particle_xs, particle_ys,
                    s=5, c='k', alpha=0.5, label="Particles")

        # Landmark ground truth locations and indexes
        landmark_xs = []
        landmark_ys = []
        for location in self.landmark_locations:
            landmark_xs.append(self.landmark_locations[location][0])
            landmark_ys.append(self.landmark_locations[location][1])
            index = self.landmark_indexes[location] + 6
            plt.text(landmark_xs[-1], landmark_ys[-1], str(index),
                     alpha=0.5, fontsize=10)
        plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2,
                    marker='*', label='Landmark Ground Truth')

        # Landmark estimated locations
        estimate_xs = []
        estimate_ys = []
        for i in range(self.N_landmarks):
            if self.landmark_observed[i]:
                estimate_xs.append(self.landmark_states[i, 0])
                estimate_ys.append(self.landmark_states[i, 1])
                plt.text(estimate_xs[-1], estimate_ys[-1],
                         str(i+6), fontsize=10)
        plt.scatter(estimate_xs, estimate_ys,
                    s=50, c='k', marker='P', label='Landmark Estimate')

        plt.title('Fast SLAM 2.0 with unknown correspondences')
        plt.legend()
        plt.xlim((-2.0, 5.5))
        plt.ylim((-7.0, 7.0))
        plt.pause(1e-16)


if __name__ == "__main__":
    pass
