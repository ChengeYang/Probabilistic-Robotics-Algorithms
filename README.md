# SLAM Algorithm Implementation - Probabilistic Robotics
#### Chenge Yang, 2019 Winter, Northwestern University
-----------------------------------------------------------------------------------------
## 1. Introduction
This project contains Python3 implementations and results of a variety of state estimation and SLAM algorithms in Sebastian Thrun's book Probabilistic Robotics using UTIAS Multi-Robot Cooperative Localization and Mapping Dataset.

As a beginner in SLAM, I always found it difficult to understand the non-intuitive mathematical equations, and I can barely find straightforward instructions on implementing these algorithms. Therefore, I created this repo to demonstrate the basic concepts behind the book, paired with results running on a simple dataset.

If you are new to SLAM problem and is reading the book Probabilistic Robotics, this repo will be perfect for you - I programmed in Python not C++ with abundant inline comments and good demonstrations of the results.

If you find anything wrong with my implementations, such as wrong understanding or code bugs, please leave a comment!

#### Table of Contents
- [1. Introduction](#1-Introduction)
- [2. Dataset](#2-Dataset)
- [3. Basic Algorithm](#3-Basic-Algorithm)
- [4. Localization](#4-Localization)
  - [4.1. EKF Localization](#41-EKF-Localization)
  - [4.2. Particle Filter Localization](#42-Particle-Filter-Localization)
- [5. EKF SLAM](#5-EKF-SLAM)
  - [5.1. EKF SLAM with Known Correspondence](#51-EKF-SLAM-with-Known-Correspondence)
  - [5.2. EKF SLAM with Unknown Correspondence](#52-EKF-SLAM-with-Unknown-Correspondence)
- [6. Graph SLAM](#6-Graph-SLAM)
  - [6.1. Graph SLAM with Known Correspondence](#61-Graph-SLAM-with-Known-Correspondence)
- [7. Fast SLAM 1](#7-Fast-SLAM-1)
  - [7.1. Fast SLAM 1 with Known Correspondence](#71-Fast-SLAM-1-with-Known-Correspondence)
  - [7.2. Fast SLAM 1 with Unknown Correspondence](#72-Fast-SLAM-1-with-Unknown-Correspondence)
- [8. Fast SLAM 2](#8-Fast-SLAM-2)
  - [8.1. Fast SLAM 2 with Unknown Correspondence](#81-Fast-SLAM-2-with-Unknown-Correspondence)
-----------------------------------------------------------------------------------------
## 2. Dataset
**UTIAS Multi-Robot Cooperative Localization and Mapping** is 2D indoor landmark-based dataset. For details please refer [here](http://asrl.utias.utoronto.ca/datasets/mrclam/index.html).

This project contains [Dataset0](0.Dataset0/) (MRSLAM_Dataset4, Robot3) and [Dataset1](1.Dataset1/) (MRCLAM_Dataset9, Robot3). All algorithms are using Dataset1 to generate the following results.

Each dataset contains five files:
* **Odometry.dat**: Control data (translation and rotation velocity)
* **Measurement.dat**: Measurement data (range and bearing data for visually observed landmarks and other robots)
* **Groundtruth.dat**: Ground truth robot position (measured via Vicon motion capture – use for assessment only)
* **Landmark_Groundtruth.dat**: Ground truth landmark positions (measured via Vicon motion capture)
* **Barcodes.dat**: Associates the barcode IDs with landmark IDs.

The way 

-----------------------------------------------------------------------------------------
## 3. Basic Algorithm

<p align = "center">
  <img src = "doc/Odometry.png" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
## 4. Localization

### 4.1. EKF Localization
<p align = "center">
  <img src = "doc/EKF_Localization_known_correspondences.png" height = "360px" style="margin:10px 10px">
</p>

### 4.2. Particle Filter Localization
<p align = "center">
  <img src = "doc/PF_Localization_known_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
## 5. EKF SLAM

### 5.1. EKF SLAM with Known Correspondence
<p align = "center">
  <img src = "doc/EKF_SLAM_known_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

### 5.2. EKF SLAM with Unknown Correspondence
<p align = "center">
  <img src = "doc/EKF_SLAM_unknown_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
## 6. Graph SLAM

### 6.1. Graph SLAM with Known Correspondence
<p align = "center">
  <img src = "doc/Graph_SLAM_known_correspondences.png" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
## 7. Fast SLAM 1

### 7.1. Fast SLAM 1 with Known Correspondence
<p align = "center">
  <img src = "doc/Fast_SLAM_1_known_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

### 7.2. Fast SLAM 1 with Unknown Correspondence
<p align = "center">
  <img src = "doc/Fast_SLAM_1_unknown_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
## 8. Fast SLAM 2

### 8.1. Fast SLAM 2 with Unknown Correspondence
<p align = "center">
  <img src = "doc/Fast_SLAM_2_unknown_correspondences.gif" height = "360px" style="margin:10px 10px">
</p>

-----------------------------------------------------------------------------------------
