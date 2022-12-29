from Simulation import log_results,C,controller,trajectory,external_forces,flying_robot,tree_dim_plot,two_dim_plot,one_dim_plot_t,plot_vel,error_plot,contr_ex_plot
import numpy as np
from scipy.integrate import solve_ivp
import functools
MAT = np.array
rot_deg = 0 # Rotation Angle on Z axis

# finish time of the simulation
tend = 100

# define initial conditions
theta = np.deg2rad(rot_deg) # Just on the Z - axis
Rba0 = MAT([[np.cos(theta),-1*np.sin(theta),0], 
            [np.sin(theta),np.cos(theta),0], 
            [0, 0, 1]])
#Rba0 = np.eye(3)
pba0 = MAT([[0,0,0]]).T
gamma0 = MAT([[0,0,0,0,0,0]]).T
vRba0 = Rba0.reshape(9,1)
e0 = np.zeros((18,1))
xi_temp = np.concatenate([pba0,vRba0,gamma0,e0])
xi0 = np.ndarray.tolist(xi_temp.T[0])

# solve the differential equations
sim = solve_ivp(flying_robot, [0, tend], xi0, max_step = 1)

# extract results
t = sim.t

# extract logs for significant time moments
traj = [trajectory.log[trajectory.t.index(time)] for time in t]
ctrl = [controller.log[controller.t.index(time)] for time in t]
extf = [external_forces.log[external_forces.t.index(time)] for time in t]
derv = [flying_robot.log[flying_robot.t.index(time)] for time in t]

n_sampl = len(t)
res_pba = MAT(sim.y)[0:3].T
res_Rba = MAT(sim.y)[3:12].reshape(3,3,n_sampl)
res_gamma = MAT(sim.y)[12:18].T
res_pbad = MAT(traj).T[0][0:3].T
res_Rbad = MAT(traj).T[0][3:12].reshape(3,3,n_sampl)
res_gammad = MAT(traj).T[0][12:18].T
res_tau = MAT(ctrl).T[0].T
res_exF = MAT(extf).T[0].T
res_tau = res_tau.T
res_exF = res_exF.T
res_dpba = MAT(derv).T[0:3].T
res_dRba = MAT(derv).T[3:12].reshape(3,3,n_sampl)
res_dgamma = MAT(derv).T[12:18].T
x, y, z, R11,R12,R13,R21,R22,R23,R31,R32,R33, u, v, w, p, q, r = sim.y[0:18]
xd, yd, zd, R11d,R12d,R13d,R21d,R22d,R23d,R31d,R32d,R33d, ud, vd, wd, pd, qd, rd = MAT(traj).T[0]

tree_dim_plot(x,y,z,xd,yd,zd)
two_dim_plot(x,y,z,xd,yd,zd)
one_dim_plot_t(t,x,y,z,xd,yd,zd)
plot_vel(t,w,v,u)
error_plot(t,x,y,z,xd,yd,zd)
contr_ex_plot(t,res_tau,res_exF)
