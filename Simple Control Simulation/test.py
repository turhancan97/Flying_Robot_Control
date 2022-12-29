# import of necessary libraries
import numpy as np # helpful with matrix operations
from scipy.integrate import solve_ivp # solving differential equations
import matplotlib.pyplot as plt # ploting results
from matplotlib import animation # make animation
from simple_control import trajectory, controller, centrifugal_and_coriolis_forces, damping_forces, Rba
from simple_control import plot_by_time,plot_planar_trajectory
############----------------###############
## Adjust the Figure Size at the beginning ##
plt.style.use('ggplot') # ggplot sytle plots
plt.rcParams["figure.figsize"] = (20,8)
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["figure.titlesize"] = 'x-large'
## plt.rcParams.keys() ## To see the plot adjustment parameters
############----------------###############


## configuration
m = 2 # mass of the flying object
Iz = 0.5
kD = np.array([[0.1], [0.5], [2]]) # # damping coefficients vectors
M = np.array([[m, 0., 0.], [0., m, 0.],[0., 0., Iz]]) # mass matrix
iM = np.linalg.inv(M) # Inverse of mass matrix
K_p = 2 # K_p for PID controller (only P controller)

## Find Tau
test_xi = np.array([[1,1,1,1,1,1]]).T
test_trajectory = trajectory(0)
tau = controller(test_xi,test_trajectory,K_p) # tau is constant

## model definition
def flat_plane(t,xi_array):
    xi = np.array([xi_array]).T
    # derivartive of x as dx
    dx = Rba(xi[2])@xi[3:]
    # derivative of velocity as acc
    acc = iM@(tau-(centrifugal_and_coriolis_forces(xi[3:],m)@xi[3:])-damping_forces(xi[3:],kD))
    dxi = np.concatenate((dx,acc))
    return np.ndarray.tolist(dxi.T[0])

## perform simulation
xi0_array = [1,1,1,1,1,1]
sim = solve_ivp(flat_plane, [0, 100], xi0_array)
t = sim.t
x, y, tta, u, v, r = sim.y
data = x, y, tta, u, v, r, t

## visualise results
plot_by_time(data)
#plt.savefig('../docs/images/measurement_by_time.png', bbox_inches='tight')
plot_planar_trajectory(data)
#plt.savefig('../docs/images/trajectories.png', bbox_inches='tight')

fig = plt.figure();
fig.set_dpi(300);

# ax = plt.axes(xlim=(np.min(x)-3, np.max(x)+3), ylim=(np.min(y)-3, np.max(y)+3))
ax = plt.axes(xlim = (-10,10), ylim = (-10,10));


def rotz(tta): 
    return np.array([[np.cos(tta), -np.sin(tta)],
                                [np.sin(tta), np.cos(tta)]])

def get_poly(x, y, tta):
    center = np.array([x, y])
    veh_tta = tta
    loc_shape = np.array([[1, 0], [-1, -1], [-1, 1]])
    loc_shape_rotated = np.r_[[rotz(veh_tta) @ v for v in loc_shape]]
    vehicle = loc_shape_rotated + center
    return vehicle


patch = plt.Polygon(get_poly(x[0], y[0], tta[0]));
line, = ax.plot([], [], lw=1);

def an_init():
    plt.plot(x[0],y[0])
    line.set_data([],[]);
    ax.add_patch(patch);
    return patch,

def animate(i):
    line.set_data(x[:i],y[:i])
    patch.set_xy(get_poly(x[i], y[i], tta[i]))
    return patch,


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=an_init,
    frames=np.shape(t)[0],
    interval=100,
    blit=False);

anim.save('../docs/images/animation.gif')
plt.show()