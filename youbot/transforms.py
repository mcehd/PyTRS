import numpy as np

# For functions that are missing from robopy


# Copyright (C) 1993-2014, by Peter I. Corke
#
# This file is part of The Robotics Toolbox for MATLAB (RTB).
# 
# RTB is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# RTB is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Leser General Public License
# along with RTB.  If not, see <http://www.gnu.org/licenses/>.
#
# http://www.petercorke.com


# Euclidean to homogeneous
# 
# H = e2h(E) is the homogeneous version (K+1xN) of the Euclidean points E (KxN) where each 
# column represents one point in RˆK.
def e2h(e: np.ndarray) -> np.ndarray:
    return np.vstack((e, np.ones((1, e.shape[1]))))

# Homogeneous to Euclidean
# 
# E = h2e(H) is the Euclidean version (K-1xN) of the homogeneous points H (KxN) where each 
# column represents one point in PˆK.
def h2e(h: np.ndarray) -> np.ndarray:
    return h[:-1, :] / h[-1, :]
    
# HOMTRANS Apply a homogeneous transformation
#
# P2 = HOMTRANS(T, P) applies homogeneous transformation T to the points 
# stored columnwise in P.
#
# - If T is in SE(2) (3x3) and
#   - P is 2xN (2D points) they are considered Euclidean (R^2)
#   - P is 3xN (2D points) they are considered projective (P^2)
# - If T is in SE(3) (4x4) and
#   - P is 3xN (3D points) they are considered Euclidean (R^3)
#   - P is 4xN (3D points) they are considered projective (P^3)
#
# TP = HOMTRANS(T, T1) applies homogeneous transformation T to the 
# homogeneous transformation T1, that is TP=T*T1.  If T1 is a 3-dimensional 
# transformation then T is applied to each plane as defined by the first two 
# dimensions, ie. if T = NxN and T=NxNxP then the result is NxNxP.
def homtrans(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    if len(p) == len(T):
        raise NotImplemented()  # TODO, if ever needed
        # if p.ndim == 3:
        #     pt = []
        #     for i in range(p.shape[2]):
        #         pt = cat(3, pt, T * p[:, :, i])
        # else:
        #     pt = T * p
    elif len(T) - len(p) == 1:
        # Second argument is Euclidean coordinate, promote to homogeneous
        pt = h2e(np.dot(T, e2h(p)))
    else:
        raise Exception('matrices and point data do not conform')
        
    return pt


# Technically this is from "common" rather than "transforms"... I'll see later if this becomes a 
# problem

# ANGDIFF Difference of two angles
#
# D = ANGDIFF(TH1, TH2) returns the difference between angles TH1 and TH2 on
# the circle.  The result is in the interval [-pi pi).  If TH1 is a column 
# vector, and TH2 a scalar then return a column vector where TH2 is modulo 
# subtracted from the corresponding elements of TH1.
#
# D = ANGDIFF(TH) returns the equivalent angle to TH in the interval [-pi pi).

def angdiff(th1, th2):
    return ((th1 - th2) + np.pi) % (2 * np.pi) - np.pi
