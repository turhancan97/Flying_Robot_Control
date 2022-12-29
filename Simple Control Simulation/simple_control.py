# import of necessary libraries
import numpy as np # helpful with matrix operations
import matplotlib.pyplot as plt # ploting results
import random, math

def centrifugal_and_coriolis_forces(vel,m):
    """
    Calculate the centrifugal and Coriolis forces matrix.
    
    Parameters
    ----------
    vel : ndarray
        A 1-dimensional array with 3 elements, representing the velocity vector [u, v, r].
    m : float
        The mass of the object.
    
    Returns
    -------
    ndarray
        A 3x3 matrix representing the centrifugal and Coriolis forces.
    
    Example
    -------
    >>> vel = np.array([1, 2, 3])
    >>> m = 1.0
    >>> centrifugal_and_coriolis_forces(vel, m)
    array([[ 0, -3,  0],
           [ 3,  0,  0],
           [ 0,  0,  0]])
    """

    r = vel[2][0]
    return np.array([[0, -r*m, 0],
                  [r*m, 0,  0],
                  [0,   0,  0]])

def damping_forces(vel, kD):
    """
    Calculate the damping forces.
    
    Parameters
    ----------
    vel : ndarray
        A 1-dimensional array with 3 elements, representing the velocity vector [u, v, r].
    kD : ndarray
        A 1-dimensional array with 3 elements, representing the damping coefficients for each velocity component.
    
    Returns
    -------
    ndarray
        A 1-dimensional array with 3 elements, representing the damping forces for each velocity component.
    
    Example
    -------
    >>> vel = np.array([1, 2, 3])
    >>> kD = np.array([0.1, 0.2, 0.3])
    >>> damping_forces(vel, kD)
    array([0.1, 0.8, 2.7])
    """

    return kD*np.sign(vel)*vel*vel

def Rba(theta):
    """
    Calculate the rotation matrix from frame `b` to frame `a`.
    
    Parameters
    ----------
    theta : float
        The angle of rotation in radians.
    
    Returns
    -------
    ndarray
        A 3x3 rotation matrix.
    
    Example
    -------
    >>> theta = np.pi / 2
    >>> Rba(theta)
    array([[ 6.123234e-17, -1.000000e+00,  0.000000e+00],
           [ 1.000000e+00,  6.123234e-17,  0.000000e+00],
           [ 0.000000e+00,  0.000000e+00,  1.000000e+00]])
    """

    return np.array([[np.cos(theta),-1*np.sin(theta),0],
           [np.sin(theta),np.cos(theta),0],
           [0, 0, 1]],dtype=object)

## trajectory generator
def trajectory(t):
    """
    Generate the desired trajectory at time `t`.
    
    Parameters
    ----------
    t : float
        The current time in seconds.
    
    Returns
    -------
    ndarray
        A 6x1 array representing the desired state at time `t`.
    """

    xd = 0
    yd = 0
    ttad = 0
    ud = 1
    vd = 0
    rd = 0
    xid = np.array([[xd,yd,ttad,ud,vd,rd]]).T
    return xid

## controller
def controller(xi, trajectory, K_p):
    """
    Calculate the control inputs for the given state and desired trajectory.
    
    Parameters
    ----------
    xi : ndarray
        A 6x1 array representing the current state.
    trajectory : ndarray
        A 6x1 array representing the desired trajectory.
    K_p : float
        The proportional gain.
    
    Returns
    -------
    ndarray
        A 3x1 array representing the control inputs.
    """

    x, y, tta, u, v, r = xi
    xd, yd, ttad, ud, vd, rd = trajectory
    taux = K_p*(ud-u)
    tauy = K_p*(vd-v)
    tautta = K_p*(rd-r)
    tau = np.array([taux,tauy,tautta])
    return tau

def plot_by_time(data):
    '''Plot the given data by time.'''
    color_list = ['b','g','r','c','m','y','k']
    data_list = ['X Position', 'Y Position', 'Angle of Rotation', 'Velocity along X-axis', 'Velocity along Y-axis', 'Velocity of Rotation']
    nrows = math.ceil((len(data)-1)/2)
    ncols = 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            try:
                axs[i, j].plot(data[6],data[k],random.choice(color_list),label=data_list[k])
                axs[i, j].legend()
            except IndexError:
                print('Problem occurred because the dataset has an odd number of features')
            finally:
                k += 1
    
    fig.supxlabel('Time [s]')
    fig.supylabel('Values')
    plt.show()

def plot_planar_trajectory(data):
    '''Plot the trajectory of given data.'''
    color_list = ['b','g','r','c','m','y','k']
    data_list = ['X Position', 'Y Position', 'Velocity along X-axis', 'Velocity along Y-axis']
    nrows = 1
    ncols = 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    axs[0].plot(data[0],data[1],random.choice(color_list),label='Trajectory by Position')
    axs[0].legend()
    axs[0].set_xlabel('X-Axis')
    axs[0].set_ylabel('Y-Axis')
    axs[1].plot(data[3],data[4],random.choice(color_list),label='Trajectory by Velocity')
    axs[1].legend()
    axs[1].set_xlabel('X-Axis')
    axs[1].set_ylabel('Y-Axis')
    
    plt.show()