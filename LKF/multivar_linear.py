import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# constants and intial parameters
num_iter = 250
dim = 4
vec_shape = (num_iter, dim)
cov_shape = (num_iter, dim, dim)
noise_ax = 9.0 # noise in unknown x accel
noise_ay = 9.0 # noise in unknown y accel
truth = np.load("truth.npy") # pre-formatted true values
data = np.load("data.npy") # pre-formatted measurements
time = data[:,0]
y = data[:,1:]
x_prime = np.zeros(vec_shape) # x prediction ws
P_prime = np.zeros(cov_shape) # P prediction ws
x = np.zeros(vec_shape) # x update ws
P = np.zeros(cov_shape) # P update ws
x[0] = y[0] # initial x
P[0] = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]]) # initial P -------------------------------------------------- this might be wrong, but LKF converges very fast, it probably wont matter...
R_LIDAR = np.array([
    [0.0225, 0.0, 0.0, 0.0],
    [0.0, 0.0225, 0.0, 0.0],
    [0.0, 0.0, 0.0450, 0.0],
    [0.0, 0.0, 0.0, 0.0450]])
H_LIDAR = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]])
f = KalmanFilter(dim_x=4, dim_z=4)
f.x = x[0]
f.R = R_LIDAR
f.H = H_LIDAR
for k in range(1, num_iter):
    d_t = time[k]-time[k-1] # time step of measurments is not consistent... ~_~
    f.F = np.array([
        [1.0, 0.0, d_t, 0.0],
        [0.0, 1.0, 0.0, d_t],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    f.Q = np.array([
        [(noise_ax**2)*(d_t**4)/4.0, 0.0, (noise_ax**2)*(d_t**3)/2.0, 0.0], 
        [0.0, (noise_ay**2)*(d_t**4)/4.0, 0.0, (noise_ay**2)*(d_t**3)/2.0], 
        [(noise_ax**2)*(d_t**3)/2.0, 0.0, (noise_ax**2)*(d_t**2)/1.0, 0.0], 
        [0.0, (noise_ay**2)*(d_t**3)/2.0, 0.0, (noise_ay**2)*(d_t**2)/1.0]])
    f.predict()
    f.update(y[k])
    x[k] = f.x

plt.clf
plt.suptitle("LKF vs true position")
plt.scatter(x[:,0], x[:,1], c='g')
plt.scatter(truth[:,0], truth[:,1], c='b', s=5)
plt.legend(["LKF results", "true values"])
plt.show()

pos_error = np.zeros(num_iter)
vel_error = np.zeros(num_iter)
for k in range(num_iter):
    diff = truth[k,:]-x[k,:]
    pos_error[k] = np.sqrt(diff[0]**2+diff[1]**2)
    vel_error[k] = np.sqrt(diff[2]**2+diff[3]**2)

plt.clf
plt.suptitle("error in LKF relative to true")
plt.scatter(np.linspace(0,num_iter, num_iter), pos_error, c='r', alpha=0.5)
plt.show()
