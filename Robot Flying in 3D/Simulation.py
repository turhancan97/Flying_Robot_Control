import numpy as np
import matplotlib.pyplot as plt
import functools

MAT = np.array
S = lambda a: MAT([[0, -a[2][0], a[1][0]],[a[2][0], 0, -a[0][0]],[-a[1][0], a[0][0], 0]]) # skew-symmetric matrix
# standard acceleration
G = 9.81 # m/s^2
# density of air
rho_air = 1.225 # kg/m^3
# mass
m = 10 # kg
# volume
V = 0.01 # m^3
# inertia tensor
I = np.eye(3, dtype=int)
# matrix of aerodynamical coefficients
Ca =  -5*np.eye(6, dtype=int)
# position of the center of gravity in the local frame
pag = MAT([[0],[0],[0]])
# position of the center of volume in the local frame
pav = pag
# mass matrix and its inversion
M = np.concatenate((np.concatenate((m*I,-m*S(pag)),axis=1),np.concatenate((m*S(pag),I),axis=1)),axis=0)
iM = np.linalg.inv(M)
# simulation(controller,trajectory,Kp,Kd)

# controller = 0 -> No Controller
# controller = 1 -> P Controller
# controller = 2 -> PD Controller

# trjectory = 0 -> Helix trajectory tracking
# trjectory = 1 -> Standing on (1,1,1) (stabilization)
# trjectory = 2 -> line Tracking

# Tracking
#Kp = [15,24,250] #-> Tracking
#Kd = [-5,5,0] # -> Tracking

# Stabilization
Kp = [25,200,1500] # -> stabilization #Kp = [25,200,35]
Kd = [30,150,500] # -> stabilization #Kd = [30,150,32.7]
traj_val = 1
cont_val = 2


def log_results(func):
    @functools.wraps(func)
    def w_dec(*args, **kwargs):
        res = func(*args, **kwargs)
        t_old = -1 if len(w_dec.t) == 0 else w_dec.t[-1]
        t_new = args[0]
        if t_new > t_old:
            w_dec.log.append(res)
            w_dec.t.append(args[0])
        else:
            f = filter(lambda x: x >= t_new, w_dec.t)
            idx = w_dec.t.index(next(f))
            w_dec.log = w_dec.log[0:idx]+[res]
            w_dec.t = w_dec.t[0:idx]+[t_new]
        return res
    w_dec.log = []
    w_dec.t = []
    return w_dec

# matrix of Coriolis and centrifugal forces
@log_results
def C(t,gamma):
    """
    Coriolis and centrifugal forces

    >>> gamma = MAT([[1,2,3,4,5,6]]).T; C(0,gamma).shape
    (6, 6)
    """
    omaa = gamma[3:6]
    ## write your code here
    C = np.concatenate((np.concatenate((m*S(omaa),-m*S(omaa)@S(pag)),axis=1),np.concatenate((m*S(pag)@S(omaa),-S(omaa)*I),axis=1)),axis=0)
    return C
    
@log_results
def controller(t, xi, xid):
    """
    Controller function.

    >>> xi = np.zeros((36,1))
    >>> xid = np.zeros((18,1)); 
    >>> controller(0,xi,xid).shape
    (6, 1)
    """
    x, y, z, R11,R12,R13,R21,R22,R23,R31,R32,R33, u, v, w, p, q, r = xi[0:18]
    iex, iey, iez, ieR11,ieR12,ieR13,ieR21,ieR22,ieR23,ieR31,ieR32,ieR33, ieu, iev, iew, iep, ieq, ier  = xi[18:36]
    pba = MAT([x,y,z])
    Rba = MAT([R11,R12,R13,R21,R22,R23,R31,R32,R33]).reshape(3,3)
    gamma = MAT([u,v,w,p,q,r])
    vaa = MAT([u,v,w])
    omaa = MAT([p,q,r])

    xd, yd, zd, R11d,R12d,R13d,R21d,R22d,R23d,R31d,R32d,R33d, ud, vd, wd, pd, qd, rd = xid
    pbad = MAT([xd,yd,zd])
    Rbad = MAT([R11d,R12d,R13d,R21d,R22d,R23d,R31d,R32d,R33d]).reshape(3,3)
    gammad = MAT([ud,vd,wd,pd,qd,rd])
    vaad = MAT([ud,vd,wd])
    omaad = MAT([pd,qd,rd])
    
    Rab = Rba.T
    
    if  cont_val == 0: # When the controller is not working
        tauu = 0
        tauv = 0 # -5
        tauw = 0 # 97.9798275
        taup = 0
        tauq = 0
        taur = 0
        tau = MAT([[tauu,tauv,tauw,taup,tauq,taur]]).T
        return tau

    if  cont_val == 1:
        #P controller
        #Kp = [15,24,250]
        P = Rab@(pbad-pba) # Pbad and Pba is in global frame so we need to translate them to local frame
        # Rba translate it to global frame and Rab translate it to global frame
        e_x = P[0][0]
        e_y = P[1][0]
        e_z = P[2][0]
        
        # No Rotation
        #e_x = xd-x
        #e_y = yd-y
        #e_z = zd-z
        
        taux = Kp[0]*(e_x) #Try Rba
        tauy = Kp[1]*(e_y)
        tauz = Kp[2]*(e_z)

        tau = MAT([[taux,tauy,tauz,0,0,0]]).T

        return tau

    if  cont_val == 2:

        #Error of Position
        P = Rab@(pbad-pba)
        e_x = P[0][0]
        e_y = P[1][0]
        e_z = P[2][0]
        
        # No Rotation
        #e_x = xd-x
        #e_y = yd-y
        #e_z = zd-z

        #Error of derivative of Position (Error of Velocity)
        D = (Rab@gammad[0:3]-gamma[0:3]) # We shouldn't rotate the gamma because from definition it is in the local frame
        
        d_e_x = D[0][0]
        d_e_y = D[1][0]
        d_e_z = D[2][0]
        
        # No Rotation
        #_e_x = ud-u
        #_e_y = vd-v
        #_e_z = wd-w
        
        #PD controller
        #Kp = [15,24,250] # 0.5,0.2,500
        P_x = Kp[0]*(e_x) 
        P_y = Kp[1]*(e_y)
        P_z = Kp[2]*(e_z)

        ###################
        #Kd = [-5,5,0]


        D_x = Kd[0]*(d_e_x)
        D_y = Kd[1]*(d_e_y)
        D_z = Kd[2]*(d_e_z)


        taux = P_x + D_x
        tauy = P_y + D_y
        tauz = P_z + D_z


        tau = MAT([[taux,tauy,tauz,0,0,0]]).T

        return tau

@log_results
def trajectory(t):
    """
    Trajectory generator function.
    pbad - desired position vector
    Rbad - desired orientation (in the form of the rotation matrix)
    gammad - desired velocieties

    >>> trajectory(0).shape
    (18, 1)
    """

    Rbad = MAT([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],])

    ud = 0
    vd = 0
    wd = 0
    pd = 0
    qd = 0
    rd = 0
    
    if traj_val == 0:
        # Helix trajectory following
        # r = radius o helix
        r = 5
        xd = r*np.sin(t)
        yd = r*np.cos(t)
        zd = t
    
    
    if traj_val == 1:
        #Standing on (1,1,1) - stabilization
        xd = 1
        yd = 1
        zd = 1

    if traj_val == 2:
        #line trajectory
        xd = 0
        yd = t
        zd = t


    pbad = MAT([[xd,yd,zd]]).T
    vRbad = Rbad.reshape(9, 1)
    gammad = MAT([[ud,vd,wd,pd,qd,rd]]).T
    xid = np.concatenate([pbad,vRbad,gammad])
    return xid
    
@log_results
def external_forces(t,Rba,gamma):
    """
    Calculation of external forces.

    >>> external_forces(0,np.eye(3),np.zeros((6,1))).shape
    (6, 1)
    """
    # vector of wind velocities expressed in global frame
    gammab_wind = MAT([[0,1,0,0,0,0]]).T  # There is a wind forces on y-axis
    
    Rab = Rba.T
    ## write your code here
    # gravity
    Fbg =  MAT([
        [0],
        [0],
        [-m*G]])
    Fag = Rab @ Fbg # Force in local frame
    Nga = S(pag) @ Fag # Tork
    # buoyancy
    Fbv = MAT([
        [0],
        [0],
        [rho_air*G*V]])
    Fav = Rab @ Fbv # Force in local frame
    Nva = S(pav) @ Fav # Tork
    # aerodynamical forces
    gamma_aw = np.concatenate((np.concatenate((Rab,np.zeros((3,3),dtype="int")),axis=1),np.concatenate((np.zeros((3,3),dtype="int"),Rab),axis=1)),axis=0)@gammab_wind
    gamma_aaw = gamma - gamma_aw
    bgamma_aw = Ca @ np.sign(gamma_aaw)*gamma_aaw*gamma_aaw
    # all forces together
    #a = np.concatenate((Fbg,Nga),axis = 1)
    #b = np.concatenate((Fbv,Nva),axis = 1)
    #c = np.concatenate((gamma_aaw,bgamma_aw),axis = 1)
    #d = np.concatenate((a,b),axis=1)
    gamma_r = np.concatenate(((Fag+Fav),(Nga+Nva)),axis = 0)
    all_external_forces = gamma_r + bgamma_aw
    #######################

    return all_external_forces
    
@log_results
def flying_robot(t,xi_array):
    """
    Equations of motion of flying robot.

    >>> xi = range(36); len(flying_robot(0,xi))
    36
    """
    xi = MAT([xi_array]).T
    x, y, z, R11,R12,R13,R21,R22,R23,R31,R32,R33, u, v, w, p, q, r = xi[0:18]
    iex, iey, iez, ieR11,ieR12,ieR13,ieR21,ieR22,ieR23,ieR31,ieR32,ieR33, ieu, iev, iew, iep, ieq, ier  = xi[18:36]
    pba = MAT([x,y,z])
    n1 = np.linalg.norm([R11,R12,R13])
    n2 = np.linalg.norm([R21,R22,R23])
    n3 = np.linalg.norm([R31,R32,R33])
    Rba = MAT([R11/n1,R12/n1,R13/n1,R21/n2,R22/n2,R23/n2,R31/n3,R32/n3,R33/n3]).reshape(3,3)

    gamma = MAT([u,v,w,p,q,r])
    vaa = MAT([u,v,w])
    omaa = MAT([p,q,r])
    
    
    ## write your code here
    # obtain trajectory
    traj = trajectory(t)
    # calculate control signal
    tau = controller(t, xi, traj)
    # apply influence of the environment
    all_external_forces = external_forces(t,Rba,gamma)
    # equations of motion
    dpba = Rba@vaa
    dRba = Rba@S(omaa)
    dgamma = iM@(tau+all_external_forces-(C(t,gamma)@gamma))
    ###############################

    vdRba = dRba.reshape(9,1)
    dxi = np.concatenate([dpba, vdRba, dgamma, traj-xi[0:18]])
    return np.ndarray.tolist(dxi.T[0])
        
def tree_dim_plot(x,y,z,xd,yd,zd):
    ## write your code here
    fig = plt.figure()
    
    if traj_val == 1:
        # desired position in 3 dimmensions
        ax = plt.axes(projection="3d")
        # actual position in 3 dimmensions
        ax.scatter(1, 1.05, 0.94,label ="Desired Point that Robot Should Reach") # Data for three-dimensional 
        ax.plot3D(x, y, z, "red",label = "Actual Trajectory of Flying Object") # Data for three-dimensional
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    else:
        # desired position in 3 dimmensions
        ax = plt.axes(projection="3d")
        # actual position in 3 dimmensions
        ax.plot3D(xd, yd, zd, "black",label ="desired trajectory") # Data for three-dimensional 
        ax.plot3D(x, y, z, "red",label = "actual trajectory") # Data for three-dimensional
        ax.legend()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    
def two_dim_plot(x,y,z,xd,yd,zd):
    if traj_val == 1:
        plt.figure()
        plt.plot(x, y,"red",label='X-Y')
        plt.scatter(1, 1.05,label="Desired Point")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best')
        plt.show()
        ####################
        plt.figure()
        plt.plot(y, z,"red", label='Y-Z Actual')
        plt.scatter(1.05, 0.94,label="Desired Point")
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.legend(loc='best')
        plt.show()
        ###################
        plt.figure()
        plt.plot(x, z,"red",label='X-Z')
        plt.scatter(1,0.94,label="Desired Point")
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend(loc='best')
        plt.show()
    
    else:      
        plt.figure()
        plt.plot(x, y, label='X-Y')
        plt.plot(xd,yd,'--',label="X-Y Desired")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best')
        plt.show()
        ####################
        plt.figure()
        plt.plot(y, z, label='Y-Z Actual')
        plt.plot(yd,zd,'--',label="Y-Z Desired")
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.legend(loc='best')
        plt.show()
        ###################
        plt.figure()
        plt.plot(x, z, label='X-Z')
        plt.plot(xd,zd,'--',label="X-Z Desired")
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend(loc='best')
        plt.show()
    
def one_dim_plot_t(t,x,y,z,xd,yd,zd):
    plt.figure()
    plt.plot(t, z,label='Z')
    plt.plot(t,zd,'--',label="Zd")
    plt.legend(loc='best')
    plt.show()
    ####################
    plt.figure()
    plt.plot(t, y, label='Y')
    plt.plot(t, yd,'--', label='Yd')
    plt.legend(loc='best')
    plt.show()
    ###################
    plt.figure()
    plt.plot(t, x, label='X')
    plt.plot(t, xd,'--', label='Xd')
    plt.legend(loc='best')
    plt.show()
    
def plot_vel(t,w,v,u):
    plt.figure()
    plt.plot(t, w, label='(W)-Linear Velocity in Z-Axis')
    plt.legend(loc='best')
    plt.show()
    ####################
    plt.figure()
    plt.plot(t, v, label='(V)-Linear Velocity in Y-Axis')
    plt.legend(loc='best')
    plt.show()
    ###################
    plt.figure()
    plt.plot(t, u,label='(U)-Linear Velocity in X-Axis')
    plt.legend(loc='best')
    plt.show()
    
    
def error_plot(t,x,y,z,xd,yd,zd):
    ## Error of Positions
    plt.figure()
    plt.plot(t, zd-z,label='Z Position Error')
    plt.legend(loc='best')
    plt.show()
    ####################
    plt.figure()
    plt.plot(t, yd-y, label='Y Position Error')
    plt.legend(loc='best')
    plt.show()
    ###################
    plt.figure()
    plt.plot(t, xd-x, label='X Position Error')
    plt.legend(loc='best')
    plt.show()

def contr_ex_plot(t,res_tau,res_exF):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(3,2,1)
    ax1.plot(t,res_tau[0],"red",label='Control Signal X - taux')
    ax1.legend(loc='best')
    ax1 = fig1.add_subplot(3,2,2)
    ax1.plot(t,res_exF[0],"red",label='External Force on X')
    ax1.legend(loc='best')
    ax1 = fig1.add_subplot(3,2,3)
    ax1.plot(t,res_tau[1],"blue",label='Control Signal Y - tauy')
    ax1.legend(loc='best')
    ax1 = fig1.add_subplot(3,2,4)
    ax1.plot(t,res_exF[1],"blue",label='External Force on Y')
    ax1.legend(loc='best')
    ax1 = fig1.add_subplot(3,2,5)
    ax1.plot(t,res_tau[2],"green",label='Control Signal Z - tauz')
    ax1.legend(loc='best')
    ax1 = fig1.add_subplot(3,2,6)
    ax1.plot(t,res_exF[2],"green",label='External Force on Z')
    ax1.legend(loc='best')