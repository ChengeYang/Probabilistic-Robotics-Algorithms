#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Measurement model for 2D Range Sensor ([range, bearing]).

Author: Chenge Yang
Email: chengeyang2019@u.northwestern.edu
'''

import numpy as np


class MeasurementModel():
    def __init__(self, Q):
        '''
        Input:
            Q: Measurement covariance matrix.
               Dimension: [2, 2].
        '''
        self.Q = Q

    def compute_expected_measurement(self, particle, landmark_idx):
        '''
        Compute the expected range and bearing given current robot state and
        landmark state.

        Measurement model: (expected measurement)
        range   =  sqrt((x_l - x_t)^2 + (y_l - y_t)^2)
        bearing  =  atan2((y_l - y_t) / (x_l - x_t)) - θ_t

        Input:
            particle: Particle() object to be updated.
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            range, bearing: the expected measurement.
        '''
        delta_x = particle.lm_mean[landmark_idx, 0] - particle.x
        delta_y = particle.lm_mean[landmark_idx, 1] - particle.y
        q = delta_x ** 2 + delta_y ** 2

        range = np.sqrt(q)
        bearing = np.arctan2(delta_y, delta_x) - particle.theta

        return range, bearing

    def compute_expected_landmark_state(self, particle, measurement):
        '''
        Compute the expected landmark location [x, y] given current robot state
        and measurement data.

        Expected landmark state: inverse of the measurement model.
        x_l = x_t + range_t * cos(bearing_t + theta_t)
        y_l = y_t + range_t * sin(bearing_t + theta_t)

        Input:
            particle: Particle() object to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
        Output:
            x, y: expected landmark state [x, y]
        '''
        x = particle.x + measurement[2] *\
            np.cos(measurement[3] + particle.theta)
        y = particle.y + measurement[2] *\
            np.sin(measurement[3] + particle.theta)

        return x, y

    def compute_landmark_jacobian(self, particle, landmark_idx):
        '''
        Computing the Jacobian wrt landmark state X_l.

        Jacobian of measurement: only take derivatives of landmark X_l.
                                 H = d h(x_t, x_l) / d (x_l)
        H_m =  delta_x/√q  delta_y/√q
               -delta_y/q  delta_x/q

        Input:
            particle: Particle() object to be updated.
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            H_m: Jacobian h'(X_t, X_l)
                 Dimension: [2, 2]
        '''
        delta_x = particle.lm_mean[landmark_idx, 0] - particle.x
        delta_y = particle.lm_mean[landmark_idx, 1] - particle.y
        q = delta_x ** 2 + delta_y ** 2

        H_1 = np.array([delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
        H_2 = np.array([-delta_y/q, delta_x/q])
        H_m = np.array([H_1, H_2])

        return H_m

    def initialize_landmark(self, particle, measurement, landmark_idx, weight):
        '''
        Initialize landmark mean and covariance for one landmark of a given
        particle.
        This landmark is the first time to be observed.
        Based on EKF method.

        Input:
            particle: Particle() object to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            landmark_idx: the index of the landmark (0 ~ 15).
            weight: the default importance factor for a new feature.
        Output:
            None.
        '''
        # Update landmark mean by inverse measurement model
        particle.lm_mean[landmark_idx] =\
            self.compute_expected_landmark_state(particle, measurement)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, landmark_idx)

        # Update landmark covariance
        H_inverse = np.linalg.inv(H_m)
        particle.lm_cov[landmark_idx] = H_inverse.dot(self.Q).dot(H_inverse.T)

        # Mark landmark as observed
        particle.lm_ob[landmark_idx] = True

        # Assign default importance weight
        particle.weight = weight

        # Update timestamp
        particle.timestamp = measurement[0]

    def landmark_update(self, particle, measurement, landmark_idx):
        '''
        Implementation for Fast SLAM 1.0.
        Update landmark mean and covariance for one landmarks of a given
        particle.
        This landmark has to be observed before.
        Based on EKF method.

        Input:
            particle: Particle() object to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            None.
        '''
        # Compute expected measurement
        range_expected, bearing_expected =\
            self.compute_expected_measurement(particle, landmark_idx)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, landmark_idx)

        # Compute Kalman gain
        Q = H_m.dot(particle.lm_cov[landmark_idx]).dot(H_m.T) + self.Q
        K = particle.lm_cov[landmark_idx].dot(H_m.T).dot(np.linalg.inv(Q))

        # Update mean
        difference = np.array([[measurement[2] - range_expected],
                               [measurement[3] - bearing_expected]])
        innovation = K.dot(difference)
        particle.lm_mean[landmark_idx] += innovation.T[0]

        # Update covariance
        particle.lm_cov[landmark_idx] =\
            (np.identity(2) - K.dot(H_m)).dot(particle.lm_cov[landmark_idx])

        # Importance factor
        particle.weight = np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                   dot(difference))[0, 0]

        # Update timestamp
        particle.timestamp = measurement[0]

    def compute_correspondence(self, particle, measurement, landmark_idx):
        '''
        Implementation for Fast SLAM 1.0.
        Compute the likelihood of correspondence for between a measurement and
        a given landmark.
        This process is the same as updating a landmark mean with EKF method.

        Input:
            particle: Particle() object to be updated.
            measurement: measurement data Z_t.
                         [timestamp, #landmark, range, bearing]
            landmark_idx: the index of the landmark (0 ~ 15).
        Output:
            likehood: likelihood of correspondence
        '''
        # Compute expected measurement
        range_expected, bearing_expected =\
            self.compute_expected_measurement(particle, landmark_idx)

        # Get Jacobian wrt landmark state
        H_m = self.compute_landmark_jacobian(particle, landmark_idx)

        Q = H_m.dot(particle.lm_cov[landmark_idx]).dot(H_m.T) + self.Q
        difference = np.array([[measurement[2] - range_expected],
                               [measurement[3] - bearing_expected]])

        # likelihood of correspondence
        likelihood = np.linalg.det(2 * np.pi * Q) ** (-0.5) *\
            np.exp(-0.5 * difference.T.dot(np.linalg.inv(Q)).
                   dot(difference))[0, 0]

        return likelihood


if __name__ == '__main__':
    pass
