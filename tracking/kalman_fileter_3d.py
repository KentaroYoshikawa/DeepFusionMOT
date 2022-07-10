# Author: wangxy
# The code refers to https://github.com/xinshuoweng/AB3DMOT

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix (状態遷移)
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function (測定)
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:,7:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.           # covariance matrix
        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01     # Process uncertainty/noise
        self.history = []
        self.still_first = True
        self.kf.x[:7] = bbox3D.reshape((7, 1))   # [x,y,z,theta,l,w,h]
        # return self.kf.x
        # return  [self.kf.x[0][0], self.kf.x[1][0], self.kf.x[2][0], self.kf.x[3][0], self.kf.x[4][0], self.kf.x[5][0], self.kf.x[6][0]]
        # self.info = info  # other info associated

    def update(self, bbox3D):
        """
        Updates the state vector with observed bbox.
        """
        self.history = []
        # if self.still_first:
        #     self.first_continuing_hit += 1  # number of continuing hit in the fist time
        # ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox3D)
        #print(bbox3D)
        #[-4.81 1.68 13.83 -2.15 4.19 1.77 2.06]

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        #print(self.kf.x)
        #[-4.81 1.68 13.83 -2.15 4.23 1.77 2.06 -0.23 -0.16 0.30]

        # return self.kf.x
        # self.info = info

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        #print(self.kf.x)=> 3Ddetection[[10] [11] [12] [13] [9] [8] [7]]??
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.history.append(self.kf.x)
        pose = self.history[-1].tolist()
        pose = np.concatenate(pose[:7], axis=0)
        #print(pose)
        #[-4.57 1.84 13.53 -2.11 4.75 1.81 1.96]

        # pose = [pose[0][0],pose[1][0],pose[2][0],pose[3][0],pose[4][0],pose[5][0],pose[6][0]]
        # return self.history[-1]
        return pose

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7,))
