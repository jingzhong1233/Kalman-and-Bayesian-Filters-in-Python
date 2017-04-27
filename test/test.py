from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

import math
import numpy as np
from numpy.random import randn
import sys

import sys
sys.path.append("..")
sys.path.append("/home/harry/Dropbox/Research/SmartDrive/Program/process")


from code1.mkf_internal import plot_track

def compute_dog_data(z_var, process_var, count=1, dt=1.):
    "returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std * dt)
        x += v*dt
        xs.append(x)
        zs.append(x + randn() * z_std)
    return np.array(xs), np.array(zs)


def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]])  # location and velocity
    kf.F = np.array([[1., dt],
                     [0., 1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])  # Measurement function
    kf.R *= R  # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P  # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf


import numpy as np

import os
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import utility as ut
import warnings

import filter.filter_lib as filter_lib
import ploter.ploter_lib as ploter_lib


warnings.filterwarnings("ignore")

# fig = plt.figure("3d plot")
# ax = fig.add_subplot(111, projection='3d')


data_root_folder = "../data/head_wear_dynamic/"
# data_root_folder="../data/2hand_round/"
folder_lst = [x[0] for x in os.walk(data_root_folder)]

file_name_lst = []
for (dirpath, dirnames, filenames) in os.walk(folder_lst[1] + "/"):
    file_name_lst.extend(filenames)
    break

all_data = []
all_label = []

for folder_name in folder_lst:
    mag_file_name_lst = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]

    count_sum = 3
    display_name_lst = []
    for item in mag_file_name_lst:
        display_name_lst.append(item.split("_")[0])

    color_index = 0
    for file_name in mag_file_name_lst:
        with open(folder_name + "/" + file_name) as csvfile:
            if (not file_name.__contains__(".csv")):
                continue
            if (not file_name.__contains__("mag")):
                continue

            # if (not file_name.__contains__("180-180") and not file_name.__contains__(
            # 		"135-135") and not file_name.__contains__("45-45") and not file_name.__contains__("0-0")):
            # 	continue
            if (not file_name.__contains__("l0r0")):
                continue

            time, raw = ut.read_input_file(csvfile)
            raw = raw - np.mean(raw[200:-1, :], axis=0)



            all_data.append(raw)

            if (file_name.__contains__("45-45") or file_name.__contains__("0-0")):
                label = "left"
            if (file_name.__contains__("135-135") or file_name.__contains__("180-180")):
                label = "right"

            # all_label.append(label)
            all_label.append(file_name)


# import pywt
from scipy.linalg import inv
for i in range(0, len(all_data)):
    seq1 = all_data[i]

    dt=0.04
    b_right=seq1[0,:]
    b_head=np.array([0,0,0])
    x=np.append(b_right,b_head)

    F=np.identity(len(x))
    P=np.identity(len(x))*1
    for ii in range(3,len(x)):
        P[ii,ii]*=3
    Q = np.zeros([len(x),len(x)])
    for ii in range(0, int(len(x)/2)):
        Q[ii, ii] = 1
    for ii in range(3, len(x)):
        Q[ii, ii] = 5

    H=np.append(np.identity(len(x)/2),np.identity(len(x)/2),axis=1)
    R=np.identity(len(x)/2)



    xs, cov = [], []
    for z in seq1:
        # predict
        x = np.dot(F, x)
        P = np.dot(F, P).dot(F.T) + Q

        # update
        S = np.dot(H, P).dot(H.T) + R
        K = np.dot(P, H.T).dot(inv(S))
        y = z - np.dot(H, x)
        x += np.dot(K, y)
        P = P - np.dot(K, H).dot(P)

        xs.append(x)
        cov.append(P)

    xs, cov = np.array(xs), np.array(cov)

    # plt.plot(xs[:, 0])
    # plt.plot(xs[:, 1])
    # plt.plot(xs[:, 2])
    #

    ploter_lib.plot_two_signal(" kalman filtered", xs[:,3:], xs[:,0:3])
    plt.show()


plt.legend()
plt.show()











dt = 1.
R_var = 10
Q_var = 0.01
x = np.array([[10.0, 4.5]]).T
P = np.diag([500, 49])
F = np.array([[1, dt],
              [0,  1]])
H = np.array([[1., 0.]])
R = np.array([[R_var]])
Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

from numpy import dot
from scipy.linalg import inv

count = 50
track, zs = compute_dog_data(R_var, Q_var, count)
xs, cov = [], []
for z in zs:
    # predict
    x = dot(F, x)
    P = dot(F, P).dot(F.T) + Q

    # update
    S = dot(H, P).dot(H.T) + R
    K = dot(P, H.T).dot(inv(S))
    y = z - dot(H, x)
    x += dot(K, y)
    P = P - dot(K, H).dot(P)

    xs.append(x)
    cov.append(P)

xs, cov = np.array(xs), np.array(cov)

plot_track(xs[:, 0], track, zs, cov, plot_P=False, dt=dt)












